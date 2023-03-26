import csv

a = [1, 2, 3, 4, 5, 6, 7]
for i in range(len(a)):
    b = open("prueba.csv", "a+", newline="")
    writer = csv.writer(b)
    writer.writerow([a[i]])
    b.close()
