/*Design 8-Queens matrix having first Queen placed. Use backtracking to place remaining
Queens to generate the final 8-queen’s matrix.*/


#include<bits/stdc++.h>
using namespace std;

const int N = 8;

void printBoard(vector<vector<int>> &board){
    for(int i=0; i<N; i++){
        for(int j=0; j<N; j++){
           if(board[i][j] == 1){
                cout << " Q ";
           } 
           else{
                cout << " * ";
           }
        }
        cout << endl;
    }
}

bool isSafe(vector<vector<int>> &board, int row, int col){

    for(int i=0; i<row; i++){
        if(board[i][col] == 1){
            return false;
        }
    }

    for(int i=row-1, j=col-1; i>=0 && j>=0; i--, j--){
        if(board[i][j] == 1){
            return false;
        }
    }

    for(int i=row-1, j=col+1; i>=0 && j<N; i--, j++){
        if(board[i][j] == 1){
            return false;
        }
    }

    return true;
}

bool solveNQueen(vector<vector<int>> &board, int row){
    if(row == N){
        return true;
    }

    // Skip row if queen already placed
    bool rowHasQueen = false;
    for(int col=0; col<N; col++){
        if(board[row][col] == 1){
            rowHasQueen = true;
            break;
        }
    }

    if(rowHasQueen) return solveNQueen(board, row+1);

    for(int col = 0; col<N; col++){
        if(board[row][col] == 1 || !isSafe(board, row, col)){
            continue;
        }
        board[row][col] = 1;

        if(solveNQueen(board, row+1)){
            return true;
        }
        board[row][col] = 0;
    }

    return false;
}


int main(){
    vector<vector<int>> board(N, vector<int>(N, 0));

    int firstRow, firstCol;
    cout << "Enter row and col of first queen (0-7): ";
    cin >> firstRow >> firstCol;

    board[firstRow][firstCol] = 1;

    // Find the first row without a queen
    int startRow = 0;
    while(startRow < N && count(board[startRow].begin(), board[startRow].end(), 1) > 0){
        startRow++;
    }

    if(solveNQueen(board, startRow)){
        cout << "\n8-Queens solution with first queen placed at (" << firstRow << ", " << firstCol << "):\n";
        printBoard(board);
    }
    else{
        cout << "No solution found for given first queen position" << endl;
    }

    return 0;
}
