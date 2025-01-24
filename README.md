Ask your Omi device anything, and get instant, AI-powered answers from Perplexity. Seamless integration, smarter searches, zero hassle!


[Entry for Omi Apps Hackathon](https://devpost.com/software/searchperplexity)


- Platform: We used Omi's Real-Time Transcript Processors to stream user inputs to our webhook. 
- Backend: A Flask app deployed on the cloud handles incoming transcripts. 
- AI Integration: Groq LLaMA 3 8B is used to detect user intent and rewrite queries. 
- Search Engine: Queries are sent to Perplexity, and results are optimized for display as notifications. 
- Notifications: Results are token-limited (~40 tokens) to fit iOS notification character limits.




