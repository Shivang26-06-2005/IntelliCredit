import { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import { ArrowRight, TrendingUp, AlertCircle } from 'lucide-react';
import Navbar from '../components/Navbar';
import Sidebar from '../components/Sidebar';
import DashboardCards from '../components/DashboardCards';
import { 
  RiskDistributionChart, 
  SectorBreakdownChart, 
  LoanExposureChart 
} from '../components/ChartCard';
import { getDashboardData } from '../services/api';

const Dashboard = () => {
  const navigate = useNavigate();
  const [sidebarOpen, setSidebarOpen] = useState(false);
  const [data, setData] = useState(null);
  const [loading, setLoading] = useState(true);

  const [error, setError] = useState(null);

  useEffect(() => {
    const fetchData = async () => {
      try {
        const result = await getDashboardData();
        setData(result);
        setError(null);
      } catch (err) {
        console.error('Failed to fetch dashboard data:', err);
        setError('Failed to load dashboard data. Please try again.');
        // Set empty data structure
        setData({
          stats: { totalApplications: 0, pendingAnalysis: 0, approvedLoans: 0, rejectedLoans: 0 },
          recentApplications: []
        });
      } finally {
        setLoading(false);
      }
    };
    fetchData();
  }, []);

  const getStatusBadge = (status) => {
    const styles = {
      'Approved': 'bg-accent/10 text-accent',
      'Under Review': 'bg-warning/10 text-warning',
      'Rejected': 'bg-danger/10 text-danger'
    };
    return styles[status] || 'bg-gray-100 text-gray-600';
  };

  const getDecisionBadge = (decision) => {
    const styles = {
      'Approved': 'bg-accent/10 text-accent',
      'Conditional Approval': 'bg-warning/10 text-warning',
      'Rejected': 'bg-danger/10 text-danger'
    };
    return styles[decision] || 'bg-gray-100 text-gray-600';
  };

  const getScoreColor = (score) => {
    if (score >= 80) return 'text-accent';
    if (score >= 60) return 'text-warning';
    return 'text-danger';
  };

  if (loading) {
    return (
      <div className="min-h-screen bg-background flex items-center justify-center">
        <div className="w-12 h-12 border-4 border-primary/20 border-t-primary rounded-full animate-spin"></div>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-background">
      <Navbar onMenuClick={() => setSidebarOpen(true)} />
      <Sidebar isOpen={sidebarOpen} onClose={() => setSidebarOpen(false)} />
      
      <main className="lg:ml-64 pt-16 min-h-screen">
        <div className="p-6">
          {/* Header */}
          <div className="mb-8">
            <h1 className="text-2xl font-bold text-primary">Dashboard</h1>
            <p className="text-gray-500 mt-1">Overview of loan applications and risk metrics</p>
          </div>

          {/* Error Display */}
          {error && (
            <div className="bg-warning/10 border border-warning/20 rounded-xl p-4 mb-6">
              <div className="flex items-center gap-3">
                <AlertCircle className="w-5 h-5 text-warning" />
                <p className="text-sm text-gray-600">{error}</p>
              </div>
            </div>
          )}

          {/* Stats Cards */}
          <DashboardCards />

          {/* Charts Section */}
          <div className="grid grid-cols-1 lg:grid-cols-3 gap-6 mt-6">
            <RiskDistributionChart />
            <SectorBreakdownChart />
            <LoanExposureChart />
          </div>

          {/* Recent Applications */}
          <div className="mt-6 bg-white rounded-xl border border-gray-100 shadow-sm overflow-hidden">
            <div className="p-6 border-b border-gray-100 flex items-center justify-between">
              <div>
                <h3 className="text-lg font-semibold text-primary">Recent Applications</h3>
                <p className="text-sm text-gray-500 mt-1">Latest loan applications requiring review</p>
              </div>
              <button 
                onClick={() => navigate('/onboarding')}
                className="flex items-center gap-2 px-4 py-2 bg-primary text-white rounded-lg hover:bg-primary-light transition-colors"
              >
                New Analysis
                <ArrowRight className="w-4 h-4" />
              </button>
            </div>

            <div className="overflow-x-auto">
              <table className="w-full">
                <thead className="bg-gray-50">
                  <tr>
                    <th className="px-6 py-4 text-left text-xs font-semibold text-gray-500 uppercase tracking-wider">Company</th>
                    <th className="px-6 py-4 text-left text-xs font-semibold text-gray-500 uppercase tracking-wider">Loan Amount</th>
                    <th className="px-6 py-4 text-left text-xs font-semibold text-gray-500 uppercase tracking-wider">Risk Score</th>
                    <th className="px-6 py-4 text-left text-xs font-semibold text-gray-500 uppercase tracking-wider">Decision</th>
                    <th className="px-6 py-4 text-left text-xs font-semibold text-gray-500 uppercase tracking-wider">Status</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-gray-100">
                  {data?.recentApplications?.map((app) => (
                    <tr key={app.id} className="hover:bg-gray-50 transition-colors">
                      <td className="px-6 py-4">
                        <div className="font-medium text-gray-900">{app.companyName}</div>
                      </td>
                      <td className="px-6 py-4 text-gray-600">{app.loanAmount}</td>
                      <td className="px-6 py-4">
                        <span className={`font-semibold ${getScoreColor(app.riskScore)}`}>
                          {app.riskScore}
                        </span>
                      </td>
                      <td className="px-6 py-4">
                        <span className={`px-3 py-1 rounded-full text-xs font-medium ${getDecisionBadge(app.decision)}`}>
                          {app.decision}
                        </span>
                      </td>
                      <td className="px-6 py-4">
                        <span className={`px-3 py-1 rounded-full text-xs font-medium ${getStatusBadge(app.status)}`}>
                          {app.status}
                        </span>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>

          {/* Quick Actions */}
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6 mt-6">
            <div className="bg-gradient-to-br from-primary to-secondary rounded-xl p-6 text-white">
              <div className="flex items-start justify-between">
                <div>
                  <h3 className="text-lg font-semibold mb-2">Start New Analysis</h3>
                  <p className="text-white/70 text-sm mb-4">Upload documents and get AI-powered credit assessment</p>
                  <button 
                    onClick={() => navigate('/onboarding')}
                    className="flex items-center gap-2 px-4 py-2 bg-white text-primary rounded-lg font-medium hover:bg-gray-100 transition-colors"
                  >
                    Get Started
                    <ArrowRight className="w-4 h-4" />
                  </button>
                </div>
                <TrendingUp className="w-12 h-12 text-white/30" />
              </div>
            </div>

            <div className="bg-white rounded-xl p-6 border border-gray-100 shadow-sm">
              <div className="flex items-start gap-4">
                <div className="w-12 h-12 bg-warning/10 rounded-xl flex items-center justify-center flex-shrink-0">
                  <AlertCircle className="w-6 h-6 text-warning" />
                </div>
                <div>
                  <h3 className="text-lg font-semibold text-primary mb-2">System Alert</h3>
                  <p className="text-gray-500 text-sm">
                    3 applications are pending review for more than 5 days. Please prioritize the analysis.
                  </p>
                </div>
              </div>
            </div>
          </div>
        </div>
      </main>
    </div>
  );
};

export default Dashboard;
