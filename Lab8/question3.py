# Implement ordinal encoding and one-hot encoding methods in Python from scratch.

colors=["red","green","blue","yellow","magenta","cyan"]
one_hot=[]
lencol=len(colors)
for i in range(lencol):
    new=[]
    for j in range(lencol):
        if i==j:
            new.append(1)
        else:
            new.append(0)
    one_hot.append(new)
print(one_hot)

grades=["A+","A","B+","B","C+","C","F"]
def gradeEncoderVin(data):
    new=[]
    for grade in data:
        if grade=="A+":
            new.append(7)
        elif grade=="A":
            new.append(6)
        elif grade=="B+":
            new.append(5)
        elif grade=="B":
            new.append(4)
        elif grade=="C+":
            new.append(3)
        elif grade=="C":
            new.append(2)
        elif grade=="F":
            new.append(1)
    return new
gradeEncoded=gradeEncoderVin(grades)
print(gradeEncoded)