import React, { useEffect, useState } from 'react';
import './App.css';
import Header from './Header';
import Home from './Home';
import About from './About';
import Dashboard from './Dashboard';

interface User {
  id: string,
  name: string,
  email: string,
  profile_pic: string,
  most_recent_syllabus?: {
      id: number,
      subject: string
  },
  most_recent_topic?: {id: number, syllabus: {id: number, subject: string}, name: string}
}

function App() {
  const defaultUser: User = {id: '', name: '', email: '', profile_pic: ''}
  const [user, setUser] = useState(defaultUser);
  useEffect(() => {
    fetch('http://localhost:5000/userinfo', {
      credentials: 'include',
    })
    .then(res => res.text())
    .then(text => {
      let new_json = JSON.parse(text);
      console.log(new_json)
      setUser(new_json);
    });
  }, []);
  
  return (
    (Object.keys(user).length != 0) ? 
    <div className="App">
      <Dashboard user={user}/>
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
