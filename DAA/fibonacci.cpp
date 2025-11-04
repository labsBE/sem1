/*Write a program to calculate Fibonacci numbers and find its step count.*/


#include <bits/stdc++.h>
using namespace std;

// Recursive function
int fibrec(int n, int &step_count) {
    step_count++; 
    if (n < 0) return -1;
    if (n <= 1) return n;
    return fibrec(n - 1, step_count) + fibrec(n - 2, step_count);
}

// Iterative function
int fibiter(int n, int &step_count) {
    step_count = 1; 
    if (n < 0) return -1;
    if (n == 0) return 0;
    if (n == 1) return 1;

    int a = 0, b = 1, c;
    for (int i = 2; i <= n; i++) {
        c = a + b;
        a = b;
        b = c;
        step_count += 3; 
    }
    return b;
}

int main() {
    int n;
    cout << "Enter the value of n: ";
    cin >> n;

    int step_count_rec = 0;
    int fib_recursive = fibrec(n, step_count_rec);
    cout << "Fibonacci number (recursive) for n=" << n << " is: " << fib_recursive << endl;
    cout << "Step count (recursive): " << step_count_rec << endl;

    int step_count_iter = 0;
    int fib_iterative = fibiter(n, step_count_iter);
    cout << "Fibonacci number (iterative) for n=" << n << " is: " << fib_iterative << endl;
    cout << "Step count (iterative): " << step_count_iter << endl;

    return 0;
}
