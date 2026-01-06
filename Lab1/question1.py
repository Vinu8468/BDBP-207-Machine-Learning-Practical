# Implement AT*A, A = [1 2 3 
#                      4 5 6 ]
A=[[1,2,3],[4,5,6]]
AT=[]
for j in range(len(A[0])):
    row=[]
    for k in range(len(A)):
        row.append(A[k][j])
    AT.append(row)

# AT=[[1,4],
#     [2,5],
#     [3,6]]
# A=[[1,2,3],
#    [4,5,6]]

ATA = []
for i in range(len(AT)):
    row = []
    for j in range(len(A[0])):
        s = 0
        for k in range(len(A)):
            s += AT[i][k] * A[k][j]
        row.append(s)
    ATA.append(row)

print(ATA)
