/*
Write a smart contract on a test network, for Bank account of a customer for following
operations:
 Deposit money
 Withdraw Money
 Show balance
*/


//SPDX-License-Identifier: UNLICENSED
pragma solidity ^0.8.0;

contract Bank{

    uint256 balance = 0;
    address public accOwner;

    constructor(){
        accOwner = msg.sender;
    }


    //Deposit
    function Deposit() public payable{
        require(accOwner == msg.sender, "You are not account owner");
        require(msg.value > 0, "Amount should be greater than 0");
        balance += msg.value;
    }


    //Withdraw
    function Withdraw() public payable {
        require(accOwner == msg.sender, "You are not account owner");
        require(msg.value > 0, "Withdraw money should be greater than 0");
        balance -= msg.value;
    }


    //Display balance
    function ShowBalance() public  view returns(uint256){
        require(accOwner == msg.sender, "You are not account owner");
        return balance;
    }

}
