import { useState, useEffect } from 'react';
import { useNavigate, useLocation } from 'react-router-dom';
import { Eye, EyeOff, Shield, Lock, Mail, AlertCircle } from 'lucide-react';

const Login = () => {
  const navigate = useNavigate();
  const location = useLocation();
  const [showPassword, setShowPassword] = useState(false);
  const [formData, setFormData] = useState({
    email: '',
    password: ''
  });
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState('');

  // Check if user is already authenticated, redirect to dashboard
  useEffect(() => {
    const isAuthenticated = localStorage.getItem('isAuthenticated') === 'true';
    if (isAuthenticated) {
      navigate('/', { replace: true });
    }
  }, [navigate]);

  const handleSubmit = async (e) => {
    e.preventDefault();
    setIsLoading(true);
    setError('');
    
    // Simulate login validation
    setTimeout(() => {
      // Demo credentials validation
      if (formData.email === 'demo@intellicredit.ai' && formData.password === 'demo123') {
        // Store authentication in localStorage
        localStorage.setItem('isAuthenticated', 'true');
        localStorage.setItem('userEmail', formData.email);
        localStorage.setItem('loginTime', new Date().toISOString());
        
        setIsLoading(false);
        
        // Redirect to the page user tried to access, or dashboard
        const from = location.state?.from?.pathname || '/';
        navigate(from, { replace: true });
      } else {
        setIsLoading(false);
        setError('Invalid email or password. Please use demo credentials.');
      }
    }, 1500);
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-primary via-primary-light to-secondary flex items-center justify-center p-4">
      {/* Background Pattern */}
      <div className="absolute inset-0 opacity-10">
        <div className="absolute top-20 left-20 w-72 h-72 bg-white rounded-full blur-3xl"></div>
        <div className="absolute bottom-20 right-20 w-96 h-96 bg-accent rounded-full blur-3xl"></div>
      </div>

      <div className="relative w-full max-w-md">
        {/* Logo Section */}
        <div className="text-center mb-8">
          <div className="inline-flex items-center justify-center w-20 h-20 bg-white rounded-2xl shadow-lg mb-4">
            <div className="w-14 h-14 bg-gradient-to-br from-primary to-secondary rounded-xl flex items-center justify-center">
              <Shield className="w-8 h-8 text-white" />
            </div>
          </div>
          <h1 className="text-3xl font-bold text-white mb-2">IntelliCredit AI</h1>
          <p className="text-white/70">AI Powered Corporate Credit Appraisal System</p>
        </div>

        {/* Login Card */}
        <div className="bg-white rounded-2xl shadow-2xl p-8">
          <div className="text-center mb-6">
            <h2 className="text-xl font-semibold text-primary">Welcome Back</h2>
            <p className="text-sm text-gray-500 mt-1">Sign in to access the credit analysis platform</p>
          </div>

          {/* Error Message */}
          {error && (
            <div className="mb-4 p-3 bg-danger/10 border border-danger/20 rounded-lg flex items-center gap-2 text-danger text-sm">
              <AlertCircle className="w-4 h-4 flex-shrink-0" />
              <span>{error}</span>
            </div>
          )}

          <form onSubmit={handleSubmit} className="space-y-5">
            {/* Email Field */}
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-2">
                Email Address
              </label>
              <div className="relative">
                <Mail className="absolute left-3 top-1/2 -translate-y-1/2 w-5 h-5 text-gray-400" />
                <input
                  type="email"
                  value={formData.email}
                  onChange={(e) => setFormData({ ...formData, email: e.target.value })}
                  placeholder="credit.officer@bank.com"
                  className="w-full pl-10 pr-4 py-3 border border-gray-200 rounded-lg focus:outline-none focus:ring-2 focus:ring-primary/20 focus:border-primary transition-all"
                  required
                />
              </div>
            </div>

            {/* Password Field */}
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-2">
                Password
              </label>
              <div className="relative">
                <Lock className="absolute left-3 top-1/2 -translate-y-1/2 w-5 h-5 text-gray-400" />
                <input
                  type={showPassword ? 'text' : 'password'}
                  value={formData.password}
                  onChange={(e) => setFormData({ ...formData, password: e.target.value })}
                  placeholder="••••••••"
                  className="w-full pl-10 pr-12 py-3 border border-gray-200 rounded-lg focus:outline-none focus:ring-2 focus:ring-primary/20 focus:border-primary transition-all"
                  required
                />
                <button
                  type="button"
                  onClick={() => setShowPassword(!showPassword)}
                  className="absolute right-3 top-1/2 -translate-y-1/2 text-gray-400 hover:text-gray-600"
                >
                  {showPassword ? <EyeOff className="w-5 h-5" /> : <Eye className="w-5 h-5" />}
                </button>
              </div>
            </div>

            {/* Remember Me & Forgot Password */}
            <div className="flex items-center justify-between text-sm">
              <label className="flex items-center gap-2 cursor-pointer">
                <input type="checkbox" className="w-4 h-4 rounded border-gray-300 text-primary focus:ring-primary" />
                <span className="text-gray-600">Remember me</span>
              </label>
              <a href="#" className="text-primary hover:text-primary-light font-medium">
                Forgot password?
              </a>
            </div>

            {/* Submit Button */}
            <button
              type="submit"
              disabled={isLoading}
              className="w-full bg-primary text-white py-3 rounded-lg font-medium hover:bg-primary-light transition-all duration-200 disabled:opacity-70 disabled:cursor-not-allowed flex items-center justify-center gap-2"
            >
              {isLoading ? (
                <>
                  <div className="w-5 h-5 border-2 border-white/30 border-t-white rounded-full animate-spin"></div>
                  Signing in...
                </>
              ) : (
                'Sign In'
              )}
            </button>
          </form>

          {/* Demo Credentials */}
          <div className="mt-6 p-4 bg-gray-50 rounded-lg">
            <p className="text-xs text-gray-500 text-center">
              <span className="font-medium">Demo Credentials:</span><br />
              Email: demo@intellicredit.ai | Password: demo123
            </p>
          </div>
        </div>

        {/* Footer */}
        <p className="text-center text-white/60 text-sm mt-6">
          Protected by enterprise-grade security
        </p>
      </div>
    </div>
  );
};

export default Login;
