def attempt_float(x):
    try:
        return float(x)
    except ValueError:
        print("Value Error")
    finally:
        print("!@")


def attempt_float2(x):
    try:
        return float(x)
    except (ValueError, TypeError):
        print("Value Error")
    finally:
        print("!@")


def exc():
    try:
        print("1")
    except:
        print("except")
    else: # try执行成功后的代码
        print("Success")
    finally:
        print("finally")


exc()
