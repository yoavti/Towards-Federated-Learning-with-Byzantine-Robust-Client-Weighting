def check_positive(value):
  return value > 0


def check_non_negative(value):
  return value >= 0


def check_proportion(value):
  return 0 <= value <= 1


def check_integer(value):
  return value.is_integer()


def create_or_validator(*validators):
  def ret(value):
    return any(validator(value) for validator in validators)
  return ret


def create_optional_validator(validator):
  def ret(value):
    return value is None or validator(value)
  return ret
