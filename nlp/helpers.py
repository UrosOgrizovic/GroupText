def abbreviate_num_to_str(num):
    if num >= 100000:
        num = round(num / 100000, 2) * 100
    elif num >= 10000:
        num = round(num / 10000, 1) * 10
    elif num >= 1000:
        num = round(num / 1000)
    else:
        return str(num)
    return f'{num}k'
