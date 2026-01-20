# solving the simple classroom example using Normal Equations.

def main():
        # 0 =(XT.X)-1 XTY theta = (X transpose * X)inverse * X transpose Y
        def XTranspose(X):
            XT=[]
            for i in range(len(X)):
                set=[]
                for j in range(len(X[i])):
                    set.append(X[j][i])
                XT.append(set)
            return XT
        print(f"printing XT : {XTranspose(Xdata)}")
        XT=XTranspose(Xdata)
        # [[1, 2], [3, 4]] to [[1, 3],[2, 4]]

        def XTransposeX(XT,X):
            XTX=[]
            for i in range(len(X)):
                col=[]
                for j in range(len(X[0])):
                    val=0
                    for k in range(len(XT[0])):
                        val=val+XT[i][k]*X[k][j]
                    col.append(val)
                XTX.append(col)
            return XTX
        print(f"printing XTX : {XTransposeX(XT,Xdata)}")
        XTX=XTransposeX(XT,Xdata)
        # [[10, 14], [14, 20]]

        def XTransposeY(XT,Y):
            XTY=[]
            for i in range(len(XT)):
                val=0
                for j in range(len(Y)):
                    val=val+XT[i][j]*Y[j]
                XTY.append(val)
            return XTY
        print(f"printing XTY: {XTransposeY(XT,Ydata)}")
        XTY=XTransposeY(XT,Ydata)
        # [38, 54]

        def deter(A):
            return (A[0][0]*A[1][1])-(A[0][1]*A[1][0])
        print(f"printing determinant of XTX : {deter(XTX)}")
        det=deter(XTX)
        # 4
        invdet=1/det
        print(f"printing inverse determinant of XTX : {invdet}")
        # 0.25


        def XTransposeXinverse(XTX,determinent):
            XTXinv=[[determinent*XTX[1][1],-determinent*XTX[0][1]],
                    [-determinent*XTX[1][0],determinent*XTX[0][0]]]
            return XTXinv
        XTXinv=XTransposeXinverse(XTX,invdet)
        print(f"printing XTXinv : {XTXinv}")
        # [[5.0, -3.5], [-3.5, 2.5]]


        def theta(XTXinv,XTY):
            theta = []
            for i in range(len(XTY)):
                val = 0
                for j in range(len(XTXinv)):
                    val = val + XTXinv[i][j] * XTY[j]
                theta.append(val)
            return theta

        print(f"theta (0): {theta(XTXinv,XTY)}")
        # [1.0, 2.0]

Xdata=[[1,2],[3,4]]
print(f"printing X : {Xdata}")
Ydata=[5,11]
print(f"printing Y : {Ydata}")

if __name__=="__main__":
    main()