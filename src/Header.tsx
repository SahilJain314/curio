import React, { useEffect, useState } from 'react';
import './Header.css';

const Header = () => {
    return(
    <header className="App-header">
        <div className="header-title">
        <h3>Curio</h3>
        </div>
        <nav>
        <ul>
            <li><a href="#home">Home</a></li>
            <li><a href="#about">About</a></li>
            <li><a href="#devs">Developers</a></li>
            <li id='tryit'><a href="http://localhost:5000/login">Try it!</a></li>
        </ul>
        </nav>
    </header>
  );
}

export default Header;