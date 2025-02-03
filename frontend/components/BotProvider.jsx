import React, { createContext, useState } from 'react';

const BotContext = createContext(); // Create a new context

const BotProvider = ({ children }) => {
  const [state, setState] = useState({ /* Initial state values */ });

  return (
    <BotContext.BotProvider value={{ state, setState }}> 
      {children}
    </BotContext.BotProvider>
  );
};

export { BotContext, BotProvider };