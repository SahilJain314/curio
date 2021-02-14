import React, { useEffect, useState } from 'react';
import './App.css';
import Header from './Header';
import Home from './Home';
import About from './About';

interface User {
  id: string,
  name: string,
  email: string,
  profile_pic: string,
}

function App() {
  const [user, setUser] = useState({});
  useEffect(() => {
    fetch('http://localhost:5000/userinfo', {
      credentials: 'include'
    })
    .then(res => res.json())
    .then(data => {
      setUser(data);
      console.log(user);
      console.log({})
    });
  }, []);
  
  return (
    (Object.keys(user).length > 0) ? 
    <div className="App">
      Hello
    </div>
    :
    <div className="App">
      <Header />
      <main>
        <Home />
        <About />
      </main>
      <footer>

      </footer>
    </div>
  );
}

export default App;
