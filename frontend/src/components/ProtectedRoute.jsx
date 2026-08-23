import { Navigate, useLocation } from 'react-router-dom';

/**
 * ProtectedRoute Component
 * 
 * Wraps protected pages and checks if user is authenticated.
 * If not authenticated, redirects to /login.
 * If authenticated, renders the children components.
 */
const ProtectedRoute = ({ children }) => {
  const location = useLocation();
  
  // Check authentication status from localStorage
  const isAuthenticated = localStorage.getItem('isAuthenticated') === 'true';
  
  // If not authenticated, redirect to login with the intended destination
  if (!isAuthenticated) {
    return <Navigate to="/login" state={{ from: location }} replace />;
  }
  
  // If authenticated, render the protected content
  return children;
};

export default ProtectedRoute;
