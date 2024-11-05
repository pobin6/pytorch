import inspect
import sys
import itertools
import random
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Set, Tuple, Union, Callable
from hypothesis import strategies as st
import torch
from pathlib import Path
import bisect
import logging

@dataclass
class ConfigField:
    name: str
    type_hint: type
    default_value: Any
    parent_class: Optional[str] = None

class ConfigFuzzer:
    def __init__(self, config_module, test_model_fn: Callable):
        """
        Initialize the config fuzzer.

        Args:
            config_module: The module containing the configs to fuzz
            test_model_fn: Function that runs a test model and returns True if successful
        """
        self.config_module = config_module
        self.test_model_fn = test_model_fn
        self.fields = self._extract_config_fields()
        self.logger = self._setup_logger()

    def _setup_logger(self) -> logging.Logger:
        logger = logging.getLogger("ConfigFuzzer")
        handler = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
        return logger

    def _extract_config_fields(self) -> List[ConfigField]:
        """Extract all config fields and their types from the config module."""
        fields = []

        for name, obj in inspect.getmembers(self.config_module):
            if inspect.isclass(obj) and not name.startswith('_'):
                class_fields = inspect.getmembers(obj)
                for field_name, field_value in class_fields:
                    if not field_name.startswith('_'):
                        type_hint = self._get_type_hint(obj, field_name)
                        fields.append(ConfigField(
                            name=f"{name}.{field_name}",
                            type_hint=type_hint,
                            default_value=field_value,
                            parent_class=name
                        ))
            elif not name.startswith('_') and not inspect.ismodule(obj):
                type_hint = self._get_type_hint(self.config_module, name)
                fields.append(ConfigField(
                    name=name,
                    type_hint=type_hint,
                    default_value=obj
                ))

        return fields

    def _get_type_hint(self, obj, name) -> type:
        """Get type hint for a field, falling back to type(default_value) if not found."""
        try:
            hints = get_type_hints(obj)
            return hints.get(name, type(getattr(obj, name)))
        except Exception:
            return type(getattr(obj, name))

    def _generate_value_for_type(self, type_hint: type) -> Any:
        """Generate a random value for a given type using Hypothesis strategies."""
        if type_hint == bool:
            return st.booleans().example()
        elif type_hint == int:
            return st.integers(min_value=-1000, max_value=1000).example()
        elif type_hint == float:
            return st.floats(min_value=-1000, max_value=1000, allow_infinity=False, allow_nan=False).example()
        elif type_hint == str:
            return st.text(alphabet=st.characters(blacklist_categories=('Cs',)), min_size=1).example()
        elif getattr(type_hint, "__origin__", None) == list:
            elem_type = type_hint.__args__[0]
            return [self._generate_value_for_type(elem_type) for _ in range(random.randint(0, 3))]
        elif getattr(type_hint, "__origin__", None) == dict:
            key_type, value_type = type_hint.__args__
            return {
                self._generate_value_for_type(key_type): self._generate_value_for_type(value_type)
                for _ in range(random.randint(0, 3))
            }
        elif type_hint == Optional[str]:
            return random.choice([None, st.text().example()])
        else:
            return None

    def _set_config(self, field: ConfigField, value: Any):
        """Set a config value in the module."""
        if field.parent_class:
            parent = getattr(self.config_module, field.parent_class)
            setattr(parent, field.name.split('.')[-1], value)
        else:
            setattr(self.config_module, field.name, value)

    def _reset_configs(self):
        """Reset all configs to their default values."""
        for field in self.fields:
            self._set_config(field, field.default_value)

    def fuzz_n_tuple(self, n: int, max_combinations: int = 1000):
        """Test every combination of n configs."""
        self.logger.info(f"Starting {n}-tuple testing")

        for combo in itertools.combinations(self.fields, n):
            values = []
            for field in combo:
                value = self._generate_value_for_type(field.type_hint)
                values.append((field, value))

            self._reset_configs()
            for field, value in values:
                self._set_config(field, value)

            try:
                success = self.test_model_fn()
                if not success:
                    self.logger.error(f"Failure with config combination:")
                    for field, value in values:
                        self.logger.error(f"{field.name} = {value}")
                    return False
            except Exception as e:
                self.logger.error(f"Exception with config combination:")
                for field, value in values:
                    self.logger.error(f"{field.name} = {value}")
                self.logger.error(f"Exception: {str(e)}")
                return False

            max_combinations -= 1
            if max_combinations <= 0:
                self.logger.info("Reached maximum combinations limit")
                break

        return True

    def fuzz_random_with_bisect(self, num_attempts: int = 100):
        """Randomly test configs and bisect to minimal failing configuration."""
        self.logger.info("Starting random testing with bisection")

        for attempt in range(num_attempts):
            self.logger.info(f"Random attempt {attempt + 1}/{num_attempts}")

            # Generate random configs
            test_configs = []
            for field in self.fields:
                if random.random() < 0.3:  # 30% chance to include each config
                    value = self._generate_value_for_type(field.type_hint)
                    test_configs.append((field, value))

            # Test the configuration
            self._reset_configs()
            for field, value in test_configs:
                self._set_config(field, value)

            try:
                success = self.test_model_fn()
                if not success:
                    self.logger.info("Found failing configuration, starting bisection")
                    minimal_failing_config = self._bisect_failing_config(test_configs)
                    self.logger.error("Minimal failing configuration:")
                    for field, value in minimal_failing_config:
                        self.logger.error(f"{field.name} = {value}")
                    return False
            except Exception as e:
                self.logger.error(f"Exception during testing: {str(e)}")
                minimal_failing_config = self._bisect_failing_config(test_configs)
                self.logger.error("Minimal failing configuration:")
                for field, value in minimal_failing_config:
                    self.logger.error(f"{field.name} = {value}")
                return False

        self.logger.info("All random tests passed")
        return True

    def _bisect_failing_config(self, failing_configs: List[Tuple[ConfigField, Any]]) -> List[Tuple[ConfigField, Any]]:
        """Bisect a failing configuration to find minimal set of configs that cause failure."""
        if len(failing_configs) <= 1:
            return failing_configs

        mid = len(failing_configs) // 2
        first_half = failing_configs[:mid]
        second_half = failing_configs[mid:]

        # Test first half
        self._reset_configs()
        for field, value in first_half:
            self._set_config(field, value)

        try:
            if not self.test_model_fn():
                return self._bisect_failing_config(first_half)
        except Exception:
            return self._bisect_failing_config(first_half)

        # Test second half
        self._reset_configs()
        for field, value in second_half:
            self._set_config(field, value)

        try:
            if not self.test_model_fn():
                return self._bisect_failing_config(second_half)
        except Exception:
            return self._bisect_failing_config(second_half)

        # If neither half fails on its own, we need both
        return failing_configs

def create_simple_test_model():
    """Create a simple test model function for demonstration."""
    def test_fn():
        try:
            model = torch.nn.Sequential(
                torch.nn.Linear(10, 10),
                torch.nn.ReLU(),
                torch.nn.Linear(10, 1)
            )

            x = torch.randn(32, 10)
            model = torch.compile(model)
            y = model(x)
            return True
        except Exception as e:
            print(f"Model test failed: {str(e)}")
            return False

    return test_fn

def main():
    # Example usage
    test_model = create_simple_test_model()
    fuzzer = ConfigFuzzer(sys.modules[__name__], test_model)

    # Test every pair of configs
    fuzzer.fuzz_n_tuple(2, max_combinations=100)

    # Test random configs with bisection
    fuzzer.fuzz_random_with_bisect(num_attempts=50)

if __name__ == "__main__":
    main()
