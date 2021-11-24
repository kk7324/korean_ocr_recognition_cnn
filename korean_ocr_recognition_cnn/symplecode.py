out=0

def compare_index(index):
### 오답 index 넣는 부분
    a_n = {1,3,5,7}
    result=0
    index=int(index)
    for i in a_n:
        if i == index:
            result= 1
    return result



x= input()
k=compare_index(x)
print(k)

