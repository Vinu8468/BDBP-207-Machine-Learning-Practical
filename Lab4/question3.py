# Implement normal equations method from scratch and compare
# your results on a simulated dataset (disease score fluctuation as
# target) and the admissions dataset  You can compare the results with
# scikit-learn and your own gradient descent implementation.
import pandas as pd

def main():
    # Convert pandas -> Python lists
    X_raw = Xdata.values.tolist()
    Y = Ydata.values.tolist()


    # Add intercept term
    X = [[1] + row for row in X_raw]   # n * (d+1)
    # adding x0 to the X which is 1


    # Transpose
    def transpose(M):
        return [[M[j][i] for j in range(len(M))] for i in range(len(M[0]))]
    # converting everything which is vertical to horizontal
    XT = transpose(X)


    # Matrix multiplication
    def matmul(A, B):
        result = []
        for i in range(len(A)):
            row = []
            for j in range(len(B[0])):
                val = 0
                for k in range(len(B)):
                    val += A[i][k] * B[k][j]
                row.append(val)
            result.append(row)
        return result

    # Matrix * Vector
    def matvec(A, v):
        result = []
        for i in range(len(A)):
            val = 0
            for j in range(len(v)):
                val += A[i][j] * v[j]
            result.append(val)
        return result


    # Matrix Inverse (without numpy)
    def inverse(M):
        n = len(M)
        A = [row[:] for row in M]  # deep copy
        I = [[1 if i == j else 0 for j in range(n)] for i in range(n)]

        for i in range(n):
            pivot = A[i][i]

            for j in range(n):
                A[i][j] /= pivot
                I[i][j] /= pivot

            for k in range(n):
                if k != i:
                    factor = A[k][i]
                    for j in range(n):
                        A[k][j] -= factor * A[i][j]
                        I[k][j] -= factor * I[i][j]
        return I


    # Normal Equation
    # 0(theta)=(XTX)-1 XTY

    XTX = matmul(XT, X)
    XTX_inv = inverse(XTX)
    XTY = matvec(XT, Y)
    theta = matvec(XTX_inv, XTY)

    print("Theta (parameters):")
    for i, t in enumerate(theta):
        print(f"theta{i} = {t}")


# Load dataset
df = pd.read_csv("simulated_data_multiple_linear_regression_for_ML.csv")

Xdata = df.drop(columns=["disease_score_fluct", "disease_score"])
Ydata = df["disease_score_fluct"]

if __name__ == "__main__":
    main()
