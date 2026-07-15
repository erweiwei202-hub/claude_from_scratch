"""A simple greeting module.

This module provides a function to print personalized greetings.
"""


def greet(name: str) -> None:
    """Print a personalized greeting message.

    Args:
        name: The name of the person to greet.
    """
    print(f"Hello, {name}!")


def main() -> None:
    """Run the main greeting logic."""
    greet("World")


if __name__ == "__main__":
    main()
