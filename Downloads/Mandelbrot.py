#Test
import numpy as np
import matplotlib.pyplot as plt
X=np.arange(-2,2,0.005)
Y=np.arange(-2,2,0.005)
grid = []
def formComplex(A,B):
    for real in A:
        for i in B:
            grid.append(complex(real,i))
formComplex(X,Y)
def Q(x,c):
	return (x**2)+c 
Mandelbrot = []
def escape(c):
	k = 0
	n=150
	while n!=0:
		k=Q(k,c)
		n-=1
		if abs(k)>2:
			break
		if abs(k)<=2 and n==0:
			Mandelbrot.append(c)
for number in grid:
	escape(number)
plt.plot([number.real for number in Mandelbrot],[number.imag for number in Mandelbrot],'k.')
plt.axis([-2,2,-2,2])
plt.show()
		