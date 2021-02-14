import React, { useEffect, useState } from 'react';
import Survey from './Survey';
import './Dashboard.css';

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
  

interface DashboardProps {
    user?: User
}

const Dashboard = (props: DashboardProps) => {

    const [hasTakenSurvey, setTakenSurvey] = useState(false);
    const [syllabi, setSyllabi] = useState([]);
    const [topics, setTopics] = useState([]);
    const [video, setVideo] = useState('');
    const [activeTopic, setActiveTopic] = useState(1);
    const [effectiveness, setEffectiveness] = useState('0');
    const [preference, setPreference] = useState('0');

    function updateTopic(topic : {id: number, syllabus: {id: number, subject: string}, name: string}) {
        setEffectiveness('0');
        setPreference('0');

        if (props.user && props.user.most_recent_topic) {
            props.user.most_recent_topic = topic;
            setActiveTopic(props.user.most_recent_topic.id);
            fetch('http://localhost:5000/video/' + props.user.most_recent_topic.id, {
                credentials: 'include',
            })
            .then(res => res.text())
            .then(text => {
                let new_json = JSON.parse(text);
                setVideo(new_json.id);
                console.log(new_json);
            });
        }
    }

    function recordAffinity() {
        const affinity = (parseFloat(effectiveness) + parseFloat(preference)) / 2;

        let formData = new FormData();
        formData.append('affinity', affinity.toString());
        formData.append('video', video);

        fetch('http://localhost:5000/interaction', {
            credentials: 'include',
            method: 'POST',
            body: formData
        })
    }

    useEffect(() => {
        // Get if the user has taken the inital survey
        fetch('http://localhost:5000/surveyresponses', {
            credentials: 'include',
          })
        .then(res => res.text())
        .then(text => {
            let new_json = JSON.parse(text);
            console.log(new_json)
            setTakenSurvey(new_json.length > 0);
            console.log(hasTakenSurvey);
        });

        // Get all the syllabus options
        fetch('http://localhost:5000/syllabus', {
            credentials: 'include',
        })
        .then(res => res.text())
        .then(text => {
            let new_json = JSON.parse(text);
            setSyllabi(new_json);
            console.log(syllabi);
        });

        // Get all the topics for the syllabus
        if (props.user) {
            let syl_id = (props.user.most_recent_syllabus) ? props.user.most_recent_syllabus.id : 1;
            fetch('http://localhost:5000/topic/' + syl_id, {
                credentials: 'include',
            })
            .then(res => res.text())
            .then(text => {
                let new_json = JSON.parse(text);
                setTopics(new_json);
            });

            if (props.user.most_recent_topic) {
                console.log('here')
                setActiveTopic(props.user.most_recent_topic.id);
                fetch('http://localhost:5000/video/' + props.user.most_recent_topic.id, {
                    credentials: 'include',
                })
                .then(res => res.text())
                .then(text => {
                    let new_json = JSON.parse(text);
                    setVideo(new_json.id);
                    console.log('Video!');
                    console.log(new_json);
                });
            }
        }
    }, [props.user]);

    if (props.user) {
        if (!hasTakenSurvey) {
            return(
                <Survey />
            );
        }
        if (!props.user.most_recent_syllabus) {
            return(
                <div>
                    <h2>What topic would you like to learn about?</h2>
                    <form action="http://localhost:5000/selectsyllabus" method="post">
                        <select name="subject" id="subject">
                            {syllabi.map((syllabus: {id: number, subject: string}, index) => (
                                <option key={index} value={syllabus.subject}>{syllabus.subject}</option>
                            ))}
                        </select>
                    </form>
                </div>
            );
        }
        // Display where the user is currently on their syllabus
        console.log('Here are topics!')
        console.log(topics)
        return(
        <div className="learning-window">
            <div className="sidebar">
                <h2>{props.user.most_recent_syllabus.subject} Syllabus</h2>
                <ul className="topic-list">
                    {topics.map((topic: {id: number, syllabus: {id: number, subject: string}, name: string}, index) => {
                        if (props.user && props.user.most_recent_topic && topic.id == props.user.most_recent_topic.id) {
                            return(
                                <li key={index} id="current-topic" onClick={() => updateTopic(topic)}>{topic.name}</li>
                            );
                        }
                        return(
                            <li key={index} onClick={() => updateTopic(topic)}>{topic.name}</li>
                        );
                    })}
                </ul>
            </div>
            <div className='video-viewer'>
                <iframe width="560" height="315" src={"https://www.youtube.com/embed/" + video} frameBorder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowFullScreen></iframe>
                <p>After watching the video...</p>
                <p>How well do you feel like you understand the content?</p>
                <div>
                Not very well<input type="range" min="-5" max="5" step="0.01" value={effectiveness} className="slider" id="effectiveness" onChange={(e) => setEffectiveness(e.target.value)} />Very Well
                </div>
                <p>How well do you feel like this video matched your preffered style of learning?</p>
                <div>
                Not very well<input type="range" min="-5" max="5" step="0.01" value={preference} className="slider" id="preference" onChange={(e) => setPreference(e.target.value)} />Very Well
                </div>
                <button onClick={() => {
                    recordAffinity();
                    updateTopic(topics[activeTopic]);
                }}>Advance to the next topic</button>
                <button onClick={() => {
                    recordAffinity();
                    setEffectiveness('0');
                    setPreference('0');
                }}>Watch another video</button>
            </div>
        </div>
        );
    }

    return(
        <div>
            Not loggged in!
        </div>
    );
}

export default Dashboard;