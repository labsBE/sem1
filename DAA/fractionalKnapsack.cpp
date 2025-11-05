/*Write a program to solve a fractional Knapsack problem using a greedy method.*/


#include <bits/stdc++.h>
using namespace std;

struct Item {
    int val;
    int wt;
    int idx;
};

// Comparison function 
bool cmp(Item a, Item b) {
    double r1 = (double)a.val / a.wt;
    double r2 = (double)b.val / b.wt;
    return r1 > r2; 
}

double fractionalKnapsack(int W, vector<Item> &items) {
    
    sort(items.begin(), items.end(), cmp);

    int n = items.size();
    double finalValue = 0.0;

    for (int i = 0; i < n; i++) {
        if(W==0) break;
        if (items[i].wt <= W) {
            
            W -= items[i].wt;
            finalValue += items[i].val;
            cout << "Item" << items[i].idx + 1 << ": " << "value = " << items[i].val << " , weight = " << items[i].wt << endl; 
        } else {
            // Take fractional part
            double fraction = (double)W / items[i].wt;
            double ValueTaken = items[i].val * fraction;
            cout << "Item" << items[i].idx + 1 << ": " << "value = " << ValueTaken << " , weight = " << items[i].wt << endl;
            finalValue += ValueTaken;
            W = 0; 
        }
    }


    cout << "\nTime Complexity: O(N log N)\n";
    cout << "Space Complexity: O(1)\n";

    return finalValue;
}

int main() {
    int n, W;
    cout << "Enter number of items: ";
    cin >> n;

    vector<Item> items(n);
    cout << "Enter value and weight of each item:\n";
    for (int i = 0; i < n; i++) {
        items[i].idx = i;
        cin >> items[i].val >> items[i].wt;
    }

    cout << "Enter capacity of knapsack: ";
    cin >> W;

    double maxValue = fractionalKnapsack(W, items);

    cout << "\nMaximum value in Knapsack = " << maxValue << endl;

    return 0;
}
