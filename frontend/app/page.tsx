import dynamic from 'next/dynamic';
import { Metadata } from 'next';

export const metadata: Metadata = {
  title: 'DEMO Teacher Report Chat',
};

const Chat = dynamic(() => import('../components/Chat'), {
  ssr: !!false,
})

export default function Home() {
  return (
    <div className="container">
      <main>
        <div className="head_container">
          <div className="row">
            <div className="col span_3">
              <a id="logo" href="https://thoughtbox.ai" data-supplied-ml-starting-dark="false" data-supplied-ml-starting="false" data-supplied-ml="false">
                {/* <img className="stnd skip-lazy default-logo dark-version" width="954" height="178" alt="Thoughtbox AI" src="https://thoughtbox.ai/wp-content/uploads/2025/01/thoughtbox-logo-wide.png" srcSet="https://thoughtbox.ai/wp-content/uploads/2025/01/thoughtbox-logo-wide.png 1x, https://thoughtbox.ai/wp-content/uploads/2025/01/thoughtbox-logo-wide.png 2x" /> */} 
                <img className="stnd skip-lazy default-logo dark-version" width="32" height="32" alt="Thoughtbox AI" src="https://thoughtbox.ai/wp-content/uploads/2025/01/cropped-thoughtbox-ico-192x192.png" />
              </a>
            </div>
            <div className="col span_9 col_last header">
              DEMO Teacher Report Chat
            </div>
          </div>
        </div>
        <Chat />
        <div className='tailMessage'>
          <p><span>Get insights into the observational data collected at </span><strong>Demo School District</strong><span>.</span></p>
        </div>
      </main>
      <footer>
      </footer>
    </div>
  );
}
