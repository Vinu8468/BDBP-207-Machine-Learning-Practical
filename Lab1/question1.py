# Implement AT*A, A = [1 2 3 
#                      4 5 6 ]

A = [
    [1, 2, 3],
    [4, 5, 6]
]

AT = []
for j in range(len(A[0])):
    row = []
    for i in range(len(A)):
        row.append(A[i][j])
    AT.append(row)

ATA = []
for i in range(len(AT)):
    row = []
    for j in range(len(A[0])):
        s = 0
        for k in range(len(A)):
            s += AT[i][k] * A[k][j]
        row.append(s)
    ATA.append(row)

for row in ATA:
    print(row)

