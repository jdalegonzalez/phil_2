import React, { useEffect, useState } from 'react';
import styles from './ChatBox.module.css';
import remarkGfm from 'remark-gfm';
import rehypeHighlight from 'rehype-highlight';
import BouncingDotsLoader from './BouncingDotsLoader';
import ReactMarkdown from 'react-markdown';

const api_url = 'http://localhost:8000'

export default function ChatBox() {
  let inputBox = null;
  let messageEnd = null;

  const [messageText, setMessageText] = useState('');
  const [receivedMessages, setMessages] = useState([]);
  const [isLoading, setIsLoading] = useState(false);
  const messageTextIsEmpty = messageText.trim().length === 0;

  const addMessage = (who, message) => {
    const hist = receivedMessages.slice(-199)
    console.log(hist)
    setMessages(prev => ([...prev, {'name': who, 'data': message}]))
  }

  useEffect(() => {
    if (!isLoading && inputBox ) inputBox.focus();
  }, [isLoading])

  const askQuestion = async (messageText) => {
    addMessage('me', messageText)
    setIsLoading(true);
    setMessageText('');
    try {
      const response = await fetch(api_url + '?q=' + messageText)
      if (!response.ok) {
        throw new Error('Failed to reach server');
      }
      const data = await response.json()
      setIsLoading(false);
      addMessage('server',data.answer)
      inputBox.focus();
    }
    catch(error) {
      console.log(error);
    }
  };

  const handleFormSubmission = (event) => {
    event.preventDefault();
    askQuestion(messageText);
  };

  const handleKeyDown = (event) => {
    if (event.keyCode !== 13 || messageTextIsEmpty) {
      return;
    }
    event.preventDefault();
    askQuestion(messageText);
  };

  const messages = receivedMessages.map((message, index) => {
    // const author = message.connectionId === ably.connection.id ? 'me' : 'other';
    const author = message.name;
    console.log(message.data)
    return (
      <span key={index} className={styles.message} data-author={author}>
        {
          author == 'me' 
          ? message.data 
          : <ReactMarkdown remarkPlugins={[remarkGfm]} rehypePlugins={[rehypeHighlight]}>{message.data}</ReactMarkdown>
        }
      </span>
    );
  });

  useEffect(() => {
    messageEnd.scrollIntoView({ behaviour: 'smooth' });
  });

  return (
    <div className={styles.chatHolder}>
      <div className={styles.chatText}>
        {messages}
        <div ref={(element) => { messageEnd = element; }} ></div>
      </div>
      <form disabled={isLoading} onSubmit={handleFormSubmission} className={"relative " + styles.form}>
        <div style={{display: isLoading ? 'block' : 'none', height:'10px', width: '100px', position: 'absolute', left:'-20px', top: '-20px'}}><BouncingDotsLoader /></div>
        <textarea
          className={"outline-none border-none " + styles.textarea}
          disabled={isLoading}
          ref={(element) => {
            inputBox = element;
          }}
          value={messageText}
          placeholder="Type a message..."
          onChange={(e) => setMessageText(e.target.value)}
          onKeyDown={handleKeyDown}
        ></textarea>
          <div className="absolute bottom-2 right-1">
            <button type="submit" disabled={messageTextIsEmpty || isLoading}
              className="send text-[var(--thisSendText)] p-2 group" 
              style={{}}>
                <svg stroke="currentColor" fill="currentColor" strokeWidth="0" viewBox="0 0 16 16" className="w-4 h-4 fill-current group-hover:stroke-1" 
                     height="1em" 
                     width="1em" 
                     xmlns="http://www.w3.org/2000/svg">
                      <path d="M15.854.146a.5.5 0 0 1 .11.54l-5.819 14.547a.75.75 0 0 1-1.329.124l-3.178-4.995L.643 7.184a.75.75 0 0 1 .124-1.33L15.314.037a.5.5 0 0 1 .54.11ZM6.636 10.07l2.761 4.338L14.13 2.576zm6.787-8.201L1.591 6.602l4.339 2.76z">
                      </path>
                </svg>
            </button>
          </div>
      </form>
    </div>
  );
}