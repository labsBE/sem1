/* To implement Huffman Encoding for data compression using a Greedy Algorithm */

#include <bits/stdc++.h>
using namespace std;

class Node {
public:
    char ch;
    int freq;
    Node *left, *right;

    Node(char ch, int freq) {
        this->ch = ch;
        this->freq = freq;
        left = right = NULL;
    }
};

// step 2: comparator for min-heap
class compared {
public:
    bool operator()(Node *l, Node *r) {
        if (l->freq == r->freq) {
            return l->ch > r->ch;
        }
        return l->freq > r->freq;
    }
};

// step 3: Huffman Encoding
class huffmanEncoding {
public:
    Node *root;
    map<char, string> mapp;

    huffmanEncoding() {
        root = NULL;
    }

    // Generate Huffman Codes recursively
    void generateCodes(Node *node, string str) {
        if (node == NULL)
            return;

        // Leaf node
        if (!node->left && !node->right) {
            mapp[node->ch] = str;
            return;
        }

        generateCodes(node->left, str + "0");
        generateCodes(node->right, str + "1");
    }

    void displayCodes() {
        if (mapp.empty()) {
            cout << "No codes generated yet. Please build the tree first.\n";
            return;
        }
        cout << "\nCharacter -> Code:\n";
        for (auto &pair : mapp) {
            cout << pair.first << " : " << pair.second << endl;
        }
    }

    void buildTree() {
        
        priority_queue<Node*, vector<Node*>, compared> minHeap;

        int n;
        cout << "Enter number of unique characters: ";
        cin >> n;

        cout << "Enter each character followed by its frequency:\n";
        for (int i = 0; i < n; i++) {
            char ch;
            int freq;
            cin >> ch >> freq;
            minHeap.push(new Node(ch, freq));
        }

        // Build Huffman Tree
        while (minHeap.size() > 1) {
            Node *left = minHeap.top(); minHeap.pop();
            Node *right = minHeap.top(); minHeap.pop();

            Node *parent = new Node('$', left->freq + right->freq);
            parent->left = left;
            parent->right = right;
            minHeap.push(parent);
        }

        root = minHeap.top();  
        mapp.clear();
        generateCodes(root, ""); 
        cout << "Huffman Tree built successfully!\n";
    }

    string encodedString(const string &str) {
        if (mapp.empty()) {
            return "Error: Build the Huffman tree first!";
        }

        string res = "";
        for (auto ch : str) {
            if (mapp.find(ch) != mapp.end()) {
                res += mapp[ch];
            } else {
                cout << "Warning: Character '" << ch << "' not in code map.\n";
            }
        }
        return res;
    }

    string decodedString(const string &encoded) {
        if (root == NULL) {
            return "Error: Build the Huffman tree first!";
        }

        string res = "";
        Node *curr = root;
        for (auto ch : encoded) {
            if (ch == '0')
                curr = curr->left;
            else if (ch == '1')
                curr = curr->right;

            if (!curr->left && !curr->right) {
                res += curr->ch;
                curr = root;
            }
        }
        return res;
    }
};

int main() {
    huffmanEncoding huff;

    while (true) {
        cout << "\n====== HUFFMAN ENCODING MENU ======\n";
        cout << "1. Build Huffman Tree\n";
        cout << "2. Display Codes\n";
        cout << "3. Encode String\n";
        cout << "4. Decode String\n";
        cout << "5. Exit\n";
        cout << "Enter your choice: ";

        int op;
        cin >> op;

        switch (op) {
            case 1:
                huff.buildTree();
                break;

            case 2:
                huff.displayCodes();
                break;

            case 3: {
                string input;
                cout << "Enter the string you want to encode: ";
                cin >> input;
                cout << "Encoded String: " << huff.encodedString(input) << endl;
                break;
            }

            case 4: {
                string encoded;
                cout << "Enter the binary string to decode: ";
                cin >> encoded;
                cout << "Decoded String: " << huff.decodedString(encoded) << endl;
                break;
            }

            case 5:
                cout << "Exiting program.\n";
                return 0;

            default:
                cout << "Invalid option! Try again.\n";
        }
    }
}
