# Can you tell whether the matrix, A = [9  -15;  -15  21] is positive definite?
import sympy as sp
A = sp.Matrix([[9,-15],[-15,21]])

print("martix A :")
print(sp.pprint(A)) # very cool

# eigenvalues
eigen = A.eigenvals()
print(f"eigenvalues :{eigen}")

eigs = list(eigen.keys())
print(f"eigenvectors :{eigs}")

if all(ev >0 for ev in eigs):
    print("A is Positive Definite")
elif all(ev >= 0 for ev in eigs):
    print("\nA is Positive Semi-Definite")
elif all(ev < 0 for ev in eigs):
    print("\nA is Negative Definite")
else:
    print("\nA is Indefinite")