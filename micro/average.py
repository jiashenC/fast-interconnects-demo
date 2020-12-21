total, count = 0, 0
with open('./_tmp.log') as f:
    for line in f.read().splitlines():
        total += int(line)
        count += 1
print(total / count)
