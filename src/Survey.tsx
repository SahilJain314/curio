import React, { useEffect, useState } from 'react';
import './Survey.css';

const Survey = () => {
    return(
        <div className="survey">
            <h2>Please fill out this survey so we can get a better understanding of your learning preferences</h2>
            <form action="http://localhost:5000/surveyrespond" method="post">
                <ol>       
                    <li>What kind of book would you like to read for fun?<br/>
                    <input required type="radio" name="q1" value="0" />
                    <label>A book with lots of pictures in it</label><br/>
                    <input required type="radio" name="q1" value="1" />
                    <label>A book with lots of words in it</label><br/>
                    <input required type="radio" name="q1" value="2" />
                    <label>A book with word searches or crossword puzzles</label><br/>
                    </li><li>When you are not sure how to spell a word, what are you most likely to do?<br/>
                    <input required type="radio" name="q2" value="0" />
                    <label>Write it down to see if it looks right</label><br/>
                    <input required type="radio" name="q2" value="1" />
                    <label>Spell it out loud to see if it sounds right</label><br/>
                    <input required type="radio" name="q2" value="2" />
                    <label>Trace the letters in the air (finger spelling)</label><br/>
                    </li><li>You're out shopping for clothes, and you're waiting in line to pay. What are you most likely to do while you are waiting?<br/>
                    <input required type="radio" name="q3" value="0" />
                    <label>Look around at other clothes on the racks</label><br/>
                    <input required type="radio" name="q3" value="1" />
                    <label>Talk to the person next to you in line</label><br/>
                    <input required type="radio" name="q3" value="2" />
                    <label>Fidget or move back and forth</label><br/>
                    </li><li>When you see the word "cat," what do you do first?<br/>
                    <input required type="radio" name="q4" value="0" />
                    <label>Picture a cat in your mind</label><br/>
                    <input required type="radio" name="q4" value="1" />
                    <label>Say the word "cat" to yourself</label><br/>
                    <input required type="radio" name="q4" value="2" />
                    <label>Think about being with a cat (petting it or hearing it purr)</label><br/>
                    </li><li>What's the best way for you to study for a test?<br/>
                    <input required type="radio" name="q5" value="0" />
                    <label>Read the book or your notes and review pictures or charts</label><br/>
                    <input required type="radio" name="q5" value="1" />
                    <label>Have someone ask you questions that you can answer out loud</label><br/>
                    <input required type="radio" name="q5" value="2" />
                    <label>Make up index cards that you can review</label><br/>
                    </li><li>What's the best way for you to learn about how something works (like a computer or a video game)?<br/>
                    <input required type="radio" name="q6" value="0" />
                    <label>Get someone to show you</label><br/>
                    <input required type="radio" name="q6" value="1" />
                    <label>Read about it or listen to someone explain it</label><br/>
                    <input required type="radio" name="q6" value="2" />
                    <label>Figure it out on your own</label><br/>
                    </li><li>If you went to a school dance, what would you be most likely to remember the next day?<br/>
                    <input required type="radio" name="q7" value="0" />
                    <label>The faces of the people who were there</label><br/>
                    <input required type="radio" name="q7" value="1" />
                    <label>The music that was played</label><br/>
                    <input required type="radio" name="q7" value="2" />
                    <label>The dance moves you did and the food you ate</label><br/>
                    </li><li>What do you find most distracting when you are trying to study?<br/>
                    <input required type="radio" name="q8" value="0" />
                    <label>People walking past you</label><br/>
                    <input required type="radio" name="q8" value="1" />
                    <label>Loud noises</label><br/>
                    <input required type="radio" name="q8" value="2" />
                    <label>An uncomfortable chair</label><br/>
                    </li><li>When you are angry, what are you most likely to do?<br/>
                    <input required type="radio" name="q9" value="0" />
                    <label>Put on your "mad" face</label><br/>
                    <input required type="radio" name="q9" value="1" />
                    <label>Yell and scream</label><br/>
                    <input required type="radio" name="q9" value="2" />
                    <label>Slam doors</label><br/>
                    </li><li>When you are happy, what are you most likely to do?<br/>
                    <input required type="radio" name="q10" value="0" />
                    <label>Smile from ear to ear</label><br/>
                    <input required type="radio" name="q10" value="1" />
                    <label>Talk up a storm</label><br/>
                    <input required type="radio" name="q10" value="2" />
                    <label>Act really hyper</label><br/>
                    </li><li>When in a new place, how do you find your way around?<br/>
                    <input required type="radio" name="q11" value="0" />
                    <label>Look for a map or directory that shows you where everything is</label><br/>
                    <input required type="radio" name="q11" value="1" />
                    <label>Ask someone for directions</label><br/>
                    <input required type="radio" name="q11" value="2" />
                    <label>Just start walking around until you find what you're looking for</label><br/>
                    </li><li>Of these three classes, which is your favorite?<br/>
                    <input required type="radio" name="q12" value="0" />
                    <label>Art class</label><br/>
                    <input required type="radio" name="q12" value="1" />
                    <label>Music class</label><br/>
                    <input required type="radio" name="q12" value="2" />
                    <label>Gym class</label><br/>
                    </li><li>When you hear a song on the radio, what are you most likely to do?<br/>
                    <input required type="radio" name="q13" value="0" />
                    <label>Picture the video that goes along with it</label><br/>
                    <input required type="radio" name="q13" value="1" />
                    <label>Sing or hum along with the music</label><br/>
                    <input required type="radio" name="q13" value="2" />
                    <label>Start dancing or tapping your foot</label><br/>
                    </li><li>What do you find most distracting when in class?<br/>
                    <input required type="radio" name="q14" value="0" />
                    <label>Lights that are too bright or too dim</label><br/>
                    <input required type="radio" name="q14" value="1" />
                    <label>Noises from the hallway or outside the building (like traffic or someone cutting the grass)</label><br/>
                    <input required type="radio" name="q14" value="2" />
                    <label>The temperature being too hot or too cold</label><br/>
                    </li><li>What do you like to do to relax?<br/>
                    <input required type="radio" name="q15" value="0" />
                    <label>Read</label><br/>
                    <input required type="radio" name="q15" value="1" />
                    <label>Listen to music</label><br/>
                    <input required type="radio" name="q15" value="2" />
                    <label>Exercise (walk, run, play sports, etc.)</label><br/>
                    </li><li>What is the best way for you to remember a friend's phone number?<br/>
                    <input required type="radio" name="q16" value="0" />
                    <label>Picture the numbers on the phone as you would dial them</label><br/>
                    <input required type="radio" name="q16" value="1" />
                    <label>Say it out loud over and over and over</label><br/>
                    <input required type="radio" name="q16" value="2" />
                    <label>Write it down or store it in your phone contact list</label><br/>
                    </li><li>If you won a game, which of these three prizes would you choose?<br/>
                    <input required type="radio" name="q17" value="0" />
                    <label>A poster for the wall</label><br/>
                    <input required type="radio" name="q17" value="1" />
                    <label>A music CD or mp3 download</label><br/>
                    <input required type="radio" name="q17" value="2" />
                    <label>A game of some kind (or a football or soccer ball, etc.)</label><br/>
                    </li><li>Which would you rather go to with a group of friends?<br/>
                    <input required type="radio" name="q18" value="0" />
                    <label>A movie</label><br/>
                    <input required type="radio" name="q18" value="1" />
                    <label>A concert</label><br/>
                    <input required type="radio" name="q18" value="2" />
                    <label>An amusement park</label><br/>
                    </li><li>What are you most likely to remember about new people you meet?<br/>
                    <input required type="radio" name="q19" value="0" />
                    <label>Their face but not their name</label><br/>
                    <input required type="radio" name="q19" value="1" />
                    <label>Their name but not their face</label><br/>
                    <input required type="radio" name="q19" value="2" />
                    <label>What you talked about with them</label><br/>
                    </li><li>When you give someone directions to your house, what are you most likely to tell them?<br/>
                    <input required type="radio" name="q20" value="0" />
                    <label>A description of building and landmarks they will pass on the way</label><br/>
                    <input required type="radio" name="q20" value="1" />
                    <label>The names of the roads or streets they will be on</label><br/>
                    <input required type="radio" name="q20" value="2" />
                    <label>Follow me—it will be easier if I just show you how to get there.</label><br/></li>
                </ol>
                <input required type="submit" value="Submit"/>
            </form>
        </div>
    );
}

export default Survey;