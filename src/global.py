a = 1
b = [2, 3]

def func():
    if a == 1:
        print("a: %d" %a)
    for i in range(4):
        if i in b:
            print("%d in list b" %i)
        else:
            print("%d not in list b" %i)

if __name__ == '__main__':
    func()