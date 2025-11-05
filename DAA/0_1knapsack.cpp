
/*Write a program to solve a 0-1 Knapsack problem using dynamic programming or branch and
bound strategy.
*/

#include<bits/stdc++.h>
using namespace std;

int knapsack(int W, vector<int> &wt, vector<int> &val, int N){

    //creating table
    vector<vector<int>> dp(N+1, vector<int>(W+1, 0));

    //build dp table
    for(int i=1; i<=N; i++){
        for(int w=1; w<=W; w++){
            if(wt[i-1] <= w){
                dp[i][w] = max(val[i-1] + dp[i-1][w - wt[i-1]], dp[i-1][w]);
            }
            else{
                dp[i][w] = dp[i-1][w];
            }
        }
    }

    int res = dp[N][W];
    int w = W;
    vector<int> Selected_items;

    for(int i=N; i>0 && res > 0; i--){
        if(res == dp[i-1][w]){
            continue;
        }
        else{
            Selected_items.push_back(i-1);
            res -= val[i-1];
            w -= wt[i-1];
        }
    }

    cout << "Items included in knapsack : " << endl;
    for(int i = Selected_items.size()-1; i>=0; i--){
        cout << Selected_items[i] << " ";
    }

    cout << "Time Complexity : O(N*W)" << endl;
    cout << "Space Complexity : O(N*W)" << endl;

    return dp[N][W];

}


int main(){
    int N, W;
    cout << "Enter the number of items: ";
    cin >> N;

    vector<int> val(N), wt(N);

    cout << "Enter the values of items: ";
    for(int i=0; i<N; i++){
        cin >> val[i];
    }

    cout << "Enter the weights of the items: ";
    for(int i=0; i<N; i++){
        cin >> wt[i];
    }


    cout << "Enter the capacity of the knapsack : " ;
    cin >> W;

    int maxVal = knapsack(W, wt, val, N);
    cout << "Maximum value that can be obatined: " << maxVal << endl;

    return 0;

}
