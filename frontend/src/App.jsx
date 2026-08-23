import { BrowserRouter as Router, Routes, Route, Navigate } from 'react-router-dom';
import ProtectedRoute from './components/ProtectedRoute';
import Login from './pages/Login';
import Dashboard from './pages/Dashboard';
import EntityOnboarding from './pages/EntityOnboarding';
import UploadDocuments from './pages/UploadDocuments';
import DocumentClassification from './pages/DocumentClassification';
import SchemaMapping from './pages/SchemaMapping';
import Processing from './pages/Processing';
import Results from './pages/Results';
import CAMReport from './pages/CAMReport';
import About from './pages/About';
import Contact from './pages/Contact';

function App() {
  return (
    <Router>
      <Routes>
        {/* Public Route - Login */}
        <Route path="/login" element={<Login />} />
        
        {/* Protected Routes - Require Authentication */}
        <Route 
          path="/" 
          element={
            <ProtectedRoute>
              <Dashboard />
            </ProtectedRoute>
          } 
        />
        <Route 
          path="/onboarding" 
          element={
            <ProtectedRoute>
              <EntityOnboarding />
            </ProtectedRoute>
          } 
        />
        <Route 
          path="/upload" 
          element={
            <ProtectedRoute>
              <UploadDocuments />
            </ProtectedRoute>
          } 
        />
        <Route 
          path="/classification" 
          element={
            <ProtectedRoute>
              <DocumentClassification />
            </ProtectedRoute>
          } 
        />
        <Route 
          path="/schema-mapping" 
          element={
            <ProtectedRoute>
              <SchemaMapping />
            </ProtectedRoute>
          } 
        />
        <Route 
          path="/processing" 
          element={
            <ProtectedRoute>
              <Processing />
            </ProtectedRoute>
          } 
        />
        <Route 
          path="/results" 
          element={
            <ProtectedRoute>
              <Results />
            </ProtectedRoute>
          } 
        />
        <Route 
          path="/cam-report" 
          element={
            <ProtectedRoute>
              <CAMReport />
            </ProtectedRoute>
          } 
        />
        <Route 
          path="/about" 
          element={
            <ProtectedRoute>
              <About />
            </ProtectedRoute>
          } 
        />
        <Route 
          path="/contact" 
          element={
            <ProtectedRoute>
              <Contact />
            </ProtectedRoute>
          } 
        />
        <Route 
          path="/reports" 
          element={
            <ProtectedRoute>
              <Dashboard />
            </ProtectedRoute>
          } 
        />
        <Route 
          path="/cam-reports" 
          element={
            <ProtectedRoute>
              <CAMReport />
            </ProtectedRoute>
          } 
        />
        
        {/* Default redirect to login */}
        <Route path="*" element={<Navigate to="/login" replace />} />
      </Routes>
    </Router>
  );
}

export default App;
