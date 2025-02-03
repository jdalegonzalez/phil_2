import React from "react";
import styles from './BouncingDots.module.css';

const BouncingDotsLoader = (props) => {
  return (
      <div className={styles['bouncing-loader']}>
        <div></div>
        <div></div>
        <div></div>
      </div>
  );
};

export default BouncingDotsLoader;