// Public frontend deployment configuration.
// Keep Firebase Admin credentials and AI provider keys on the backend only.

window.COURSEMATE_API_URL = "https://course-mate-mitt.onrender.com";

// Replace these values with the public Firebase Web App configuration
// before enabling Google sign-in on the deployed frontend.
window.COURSEMATE_FIREBASE_CONFIG = {
    apiKey: "replace-with-your-firebase-web-api-key",
    authDomain: "your-project.firebaseapp.com",
    projectId: "your-project-id",
    storageBucket: "your-project.firebasestorage.app",
    messagingSenderId: "replace-with-your-sender-id",
    appId: "replace-with-your-web-app-id",
    measurementId: "replace-with-your-measurement-id"
};
