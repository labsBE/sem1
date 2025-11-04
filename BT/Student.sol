/*
Write a program in solidity to create Student data. Use the following constructs:
 Structures
 Arrays
 Fallback
Deploy this as smart contract on Ethereum and Observe the transaction fee and Gas values
*/


//SPDX-License-Identifier: UNLICENSED
pragma solidity ^0.8.0;

contract StudentInfo{


    //Structure
    struct Student{
        string name;
        uint256 rollno;
    }

    //Array
    Student[] public studentArr;

    function addStudent(string memory name, uint rollno) public {
        for(uint i=0; i<studentArr.length; i++){
            if(studentArr[i].rollno==rollno){
                revert("Student with this number already exist.");
            }
        }
        studentArr.push(Student(name, rollno));
    }

    function getStudentsLength() public view returns(uint256){
        return studentArr.length;
    }

    function displayAllStudents() public view returns(Student[] memory){
        return studentArr;
    }


    function getStudentByIndex(uint idx) public view returns(Student memory){
        require(idx < studentArr.length, "Index out of bound !!!");
        return studentArr[idx];
    }


    //Fallback

    fallback() external payable {
        //This function will handle external function calls that is not in our contract
    }

    receive() external payable {
        //This function will handle the ether sent by external user but without data mentioned
    }
}

