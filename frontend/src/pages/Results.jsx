import { useState, useEffect } from 'react';
import { useNavigate, useLocation } from 'react-router-dom';
import { 
  ArrowLeft, 
  CheckCircle, 
  AlertTriangle, 
  XCircle,
  Download,
  FileText,
  TrendingUp,
  TrendingDown,
  Brain,
  Globe,
  Scale,
  Newspaper,
  User,
  Award,
  Lightbulb,
  Info,
  ChevronRight,
  Shield,
  AlertCircle
} from 'lucide-react';
import Navbar from '../components/Navbar';
import Sidebar from '../components/Sidebar';
import RiskScoreCard from '../components/RiskScoreCard';
import { RiskBreakdownChart } from '../components/ChartCard';
import { getRiskAssessment, getResearchFindings } from '../services/api';

const Results = () => {
  const navigate = useNavigate();
  const location = useLocation();
  const effectiveLoanId = location.state?.loanId || localStorage.getItem('last_loan_id');
  const [sidebarOpen, setSidebarOpen] = useState(false);
  const [data, setData] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    const fetchResults = async () => {
      try {
        setLoading(true);
        setError(null);
        
        // Get loanId from navigation state
        const loanId = effectiveLoanId;
        
        if (loanId) {
          localStorage.setItem('last_loan_id', loanId);
          // Fetch real data from backend
          const riskData = await getRiskAssessment(loanId);
          
          // Transform backend data to frontend format
          const transformedData = {
            companyName: location.state?.entityData?.company_name || 'Unknown Company',
            riskScore: riskData.risk_score || 0,
            decision: riskData.recommendation?.replace('_', ' ') || 'Pending',
            loanLimit: '₹6 Crore', // This would come from backend
            interestRate: '12.5%',
            confidence: riskData.probability_of_default ? Math.round((1 - riskData.probability_of_default) * 100) : 78,
            positiveSignals: riskData.rule_flags?.filter(r => r.severity === 'good')?.map(r => r.description) || [
              'Consistent GST revenue',
              'Strong sector demand'
            ],
            negativeSignals: riskData.rule_flags?.filter(r => r.severity !== 'good')?.map(r => r.description) || [
              'High debt ratio detected'
            ],
            riskBreakdown: {
              financialStrength: 75,
              cashflowStability: 70,
              litigationRisk: 40,
              sectorOutlook: 80,
              promoterReputation: 60
            },
            secondaryResearch: {
              newsSentiment: { score: 65, label: 'Neutral', summary: 'No data available' },
              litigationRecords: { count: 0, details: [] },
              sectorOutlook: { score: 70, label: 'Neutral', summary: 'No data available' },
              promoterReputation: { score: 70, label: 'Good', summary: 'No data available' },
              creditRatings: {},
              recentNews: []
            },
            explainableAI: {
              riskScore: riskData.risk_score || 0,
              decision: riskData.recommendation?.replace('_', ' ') || 'Pending',
              confidence: riskData.probability_of_default ? Math.round((1 - riskData.probability_of_default) * 100) : 78,
              reasoning: {
                positiveFactors: riskData.rule_flags?.filter(r => r.severity === 'good')?.map(r => ({
                  factor: r.description,
                  impact: `+${r.penalty || 5} points`,
                  description: `Good performance on ${r.key}`
                })) || [{ factor: 'No positive factors', impact: '0 points', description: 'No data available' }],
                negativeFactors: riskData.rule_flags?.filter(r => r.severity !== 'good')?.map(r => ({
                  factor: r.description,
                  impact: `-${r.penalty || 5} points`,
                  description: `Risk detected: ${r.key}`
                })) || [{ factor: 'No negative factors', impact: '0 points', description: 'No data available' }]
              },
              recommendations: riskData.conditions || ['No recommendations available']
            }
          };
          
          setData(transformedData);
        } else {
          // No loanId, show error
          setError('No loan ID found. Please complete the analysis workflow.');
        }
      } catch (err) {
        console.error('Failed to fetch results:', err);
        setError('Failed to load analysis results: ' + (err.message || 'Unknown error'));
      } finally {
        setLoading(false);
      }
    };
    
    fetchResults();
  }, [location.state]);

  if (loading) {
    return (
      <div className="min-h-screen bg-background flex items-center justify-center">
        <div className="w-12 h-12 border-4 border-primary/20 border-t-primary rounded-full animate-spin"></div>
      </div>
    );
  }

  if (error && !data) {
    return (
      <div className="min-h-screen bg-background">
        <Navbar onMenuClick={() => setSidebarOpen(true)} />
        <Sidebar isOpen={sidebarOpen} onClose={() => setSidebarOpen(false)} />
        <main className="lg:ml-64 pt-16 min-h-screen">
          <div className="p-6 flex flex-col items-center justify-center min-h-[60vh]">
            <AlertCircle className="w-16 h-16 text-danger mb-4" />
            <h1 className="text-2xl font-bold text-primary mb-2">Error Loading Results</h1>
            <p className="text-gray-500 mb-6">{error}</p>
            <button
              onClick={() => navigate('/upload')}
              className="px-6 py-3 bg-primary text-white rounded-lg hover:bg-primary-light transition-colors"
            >
              Go Back to Upload
            </button>
          </div>
        </main>
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
          <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4 mb-8">
            <div className="flex items-center gap-4">
              <button 
                onClick={() => navigate('/')}
                className="p-2 hover:bg-gray-100 rounded-lg transition-colors"
              >
                <ArrowLeft className="w-5 h-5 text-gray-600" />
              </button>
              <div>
                <h1 className="text-2xl font-bold text-primary">Risk Analysis Results</h1>
                <p className="text-gray-500 mt-1">AI-powered credit assessment for {data?.companyName}</p>
              </div>
            </div>
            <button
              onClick={() => navigate('/cam-report', { state: { loanId: effectiveLoanId } })}
              disabled={!effectiveLoanId}
              className="flex items-center gap-2 px-4 py-2 bg-primary text-white rounded-lg hover:bg-primary-light transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
            >
              <FileText className="w-4 h-4" />
              View CAM Report
            </button>
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

          <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
            {/* Risk Score Card */}
            <div className="lg:col-span-1">
              <RiskScoreCard 
                score={data?.riskScore}
                decision={data?.decision}
                loanLimit={data?.loanLimit}
                interestRate={data?.interestRate}
              />
            </div>

            {/* Risk Breakdown Chart */}
            <div className="lg:col-span-2">
              <RiskBreakdownChart data={data?.riskBreakdown} />
            </div>
          </div>

          {/* Signals Section */}
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6 mt-6">
            {/* Positive Signals */}
            <div className="bg-white rounded-xl border border-gray-100 shadow-sm p-6">
              <div className="flex items-center gap-3 mb-4">
                <div className="w-10 h-10 bg-accent/10 rounded-lg flex items-center justify-center">
                  <CheckCircle className="w-5 h-5 text-accent" />
                </div>
                <h3 className="text-lg font-semibold text-primary">Positive Signals</h3>
              </div>
              <ul className="space-y-3">
                {data?.positiveSignals?.map((signal, index) => (
                  <li key={index} className="flex items-start gap-3">
                    <TrendingUp className="w-4 h-4 text-accent mt-0.5 flex-shrink-0" />
                    <span className="text-gray-600">{signal}</span>
                  </li>
                ))}
              </ul>
            </div>

            {/* Negative Signals */}
            <div className="bg-white rounded-xl border border-gray-100 shadow-sm p-6">
              <div className="flex items-center gap-3 mb-4">
                <div className="w-10 h-10 bg-danger/10 rounded-lg flex items-center justify-center">
                  <AlertTriangle className="w-5 h-5 text-danger" />
                </div>
                <h3 className="text-lg font-semibold text-primary">Risk Factors</h3>
              </div>
              <ul className="space-y-3">
                {data?.negativeSignals?.map((signal, index) => (
                  <li key={index} className="flex items-start gap-3">
                    <TrendingDown className="w-4 h-4 text-danger mt-0.5 flex-shrink-0" />
                    <span className="text-gray-600">{signal}</span>
                  </li>
                ))}
              </ul>
            </div>
          </div>

          {/* Secondary Research Intelligence */}
          <div className="mt-6 bg-white rounded-xl border border-gray-100 shadow-sm p-6">
            <div className="flex items-center gap-3 mb-6">
              <div className="w-10 h-10 bg-primary/10 rounded-lg flex items-center justify-center">
                <Globe className="w-5 h-5 text-primary" />
              </div>
              <div>
                <h3 className="text-lg font-semibold text-primary">Secondary Intelligence Insights</h3>
                <p className="text-sm text-gray-500">External data sources and market intelligence</p>
              </div>
            </div>

            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
              {/* News Sentiment */}
              <div className="bg-gray-50 rounded-xl p-4">
                <div className="flex items-center gap-2 mb-3">
                  <Newspaper className="w-4 h-4 text-primary" />
                  <span className="text-sm font-medium text-gray-700">News Sentiment</span>
                </div>
                <div className="flex items-center gap-2">
                  <span className={`text-lg font-bold ${
                    data?.secondaryResearch?.newsSentiment?.score >= 60 ? 'text-accent' : 
                    data?.secondaryResearch?.newsSentiment?.score >= 40 ? 'text-warning' : 'text-danger'
                  }`}>
                    {data?.secondaryResearch?.newsSentiment?.label}
                  </span>
                  <span className="text-sm text-gray-500">({data?.secondaryResearch?.newsSentiment?.score}/100)</span>
                </div>
                <p className="text-xs text-gray-500 mt-2">{data?.secondaryResearch?.newsSentiment?.summary}</p>
              </div>

              {/* Litigation Records */}
              <div className="bg-gray-50 rounded-xl p-4">
                <div className="flex items-center gap-2 mb-3">
                  <Scale className="w-4 h-4 text-primary" />
                  <span className="text-sm font-medium text-gray-700">Litigation Records</span>
                </div>
                <div className="flex items-center gap-2">
                  <span className={`text-lg font-bold ${
                    data?.secondaryResearch?.litigationRecords?.count === 0 ? 'text-accent' : 
                    data?.secondaryResearch?.litigationRecords?.count <= 2 ? 'text-warning' : 'text-danger'
                  }`}>
                    {data?.secondaryResearch?.litigationRecords?.count} Cases
                  </span>
                </div>
                <div className="mt-2 space-y-1">
                  {data?.secondaryResearch?.litigationRecords?.details?.map((record, idx) => (
                    <div key={idx} className="flex items-center gap-1 text-xs">
                      <span className={`w-2 h-2 rounded-full ${
                        record.status === 'Resolved' ? 'bg-accent' : 'bg-warning'
                      }`} />
                      <span className="text-gray-600">{record.type}: {record.status}</span>
                    </div>
                  ))}
                </div>
              </div>

              {/* Sector Outlook */}
              <div className="bg-gray-50 rounded-xl p-4">
                <div className="flex items-center gap-2 mb-3">
                  <TrendingUp className="w-4 h-4 text-primary" />
                  <span className="text-sm font-medium text-gray-700">Sector Outlook</span>
                </div>
                <div className="flex items-center gap-2">
                  <span className={`text-lg font-bold ${
                    data?.secondaryResearch?.sectorOutlook?.score >= 70 ? 'text-accent' : 
                    data?.secondaryResearch?.sectorOutlook?.score >= 50 ? 'text-warning' : 'text-danger'
                  }`}>
                    {data?.secondaryResearch?.sectorOutlook?.label}
                  </span>
                  <span className="text-sm text-gray-500">({data?.secondaryResearch?.sectorOutlook?.score}/100)</span>
                </div>
                <p className="text-xs text-gray-500 mt-2">{data?.secondaryResearch?.sectorOutlook?.summary}</p>
              </div>

              {/* Promoter Reputation */}
              <div className="bg-gray-50 rounded-xl p-4">
                <div className="flex items-center gap-2 mb-3">
                  <User className="w-4 h-4 text-primary" />
                  <span className="text-sm font-medium text-gray-700">Promoter Reputation</span>
                </div>
                <div className="flex items-center gap-2">
                  <span className={`text-lg font-bold ${
                    data?.secondaryResearch?.promoterReputation?.score >= 70 ? 'text-accent' : 
                    data?.secondaryResearch?.promoterReputation?.score >= 50 ? 'text-warning' : 'text-danger'
                  }`}>
                    {data?.secondaryResearch?.promoterReputation?.label}
                  </span>
                  <span className="text-sm text-gray-500">({data?.secondaryResearch?.promoterReputation?.score}/100)</span>
                </div>
                <p className="text-xs text-gray-500 mt-2">{data?.secondaryResearch?.promoterReputation?.summary}</p>
              </div>
            </div>

            {/* Credit Ratings */}
            <div className="mt-4 flex items-center gap-4 p-4 bg-gradient-to-r from-primary/5 to-secondary/5 rounded-xl">
              <Shield className="w-8 h-8 text-primary" />
              <div>
                <p className="text-sm font-medium text-gray-700">External Credit Ratings</p>
                <div className="flex items-center gap-4 mt-1">
                  {Object.entries(data?.secondaryResearch?.creditRatings || {}).map(([agency, rating]) => (
                    <div key={agency} className="text-center">
                      <span className="text-xs text-gray-500 uppercase">{agency}</span>
                      <p className="font-bold text-primary">{rating}</p>
                    </div>
                  ))}
                </div>
              </div>
            </div>
          </div>

          {/* Explainable AI - Reasoning Engine */}
          <div className="mt-6 bg-white rounded-xl border border-gray-100 shadow-sm p-6">
            <div className="flex items-center gap-3 mb-6">
              <div className="w-10 h-10 bg-primary/10 rounded-lg flex items-center justify-center">
                <Brain className="w-5 h-5 text-primary" />
              </div>
              <div>
                <h3 className="text-lg font-semibold text-primary">Reasoning Engine</h3>
                <p className="text-sm text-gray-500">Explainable AI decision factors</p>
              </div>
              <div className="ml-auto flex items-center gap-2">
                <span className="text-sm text-gray-500">AI Confidence:</span>
                <span className="font-bold text-primary">{data?.explainableAI?.confidence}%</span>
              </div>
            </div>

            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
              {/* Positive Factors */}
              <div>
                <h4 className="font-medium text-accent mb-4 flex items-center gap-2">
                  <TrendingUp className="w-4 h-4" />
                  Positive Contributing Factors
                </h4>
                <div className="space-y-3">
                  {data?.explainableAI?.reasoning?.positiveFactors?.map((factor, index) => (
                    <div key={index} className="flex items-start gap-3 p-3 bg-accent/5 rounded-lg border border-accent/10">
                      <div className="w-6 h-6 bg-accent/10 rounded-full flex items-center justify-center flex-shrink-0">
                        <span className="text-xs font-bold text-accent">+</span>
                      </div>
                      <div className="flex-1">
                        <div className="flex items-center justify-between">
                          <span className="font-medium text-gray-900">{factor.factor}</span>
                          <span className="text-sm font-bold text-accent">{factor.impact}</span>
                        </div>
                        <p className="text-sm text-gray-600 mt-1">{factor.description}</p>
                      </div>
                    </div>
                  ))}
                </div>
              </div>

              {/* Negative Factors */}
              <div>
                <h4 className="font-medium text-danger mb-4 flex items-center gap-2">
                  <TrendingDown className="w-4 h-4" />
                  Risk Factors
                </h4>
                <div className="space-y-3">
                  {data?.explainableAI?.reasoning?.negativeFactors?.map((factor, index) => (
                    <div key={index} className="flex items-start gap-3 p-3 bg-danger/5 rounded-lg border border-danger/10">
                      <div className="w-6 h-6 bg-danger/10 rounded-full flex items-center justify-center flex-shrink-0">
                        <span className="text-xs font-bold text-danger">-</span>
                      </div>
                      <div className="flex-1">
                        <div className="flex items-center justify-between">
                          <span className="font-medium text-gray-900">{factor.factor}</span>
                          <span className="text-sm font-bold text-danger">{factor.impact}</span>
                        </div>
                        <p className="text-sm text-gray-600 mt-1">{factor.description}</p>
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            </div>

            {/* AI Recommendations */}
            <div className="mt-6 p-4 bg-primary/5 rounded-xl border border-primary/10">
              <h4 className="font-medium text-primary mb-3 flex items-center gap-2">
                <Lightbulb className="w-4 h-4" />
                AI Recommendations
              </h4>
              <ul className="space-y-2">
                {data?.explainableAI?.recommendations?.map((rec, index) => (
                  <li key={index} className="flex items-start gap-2 text-sm text-gray-700">
                    <ChevronRight className="w-4 h-4 text-primary mt-0.5 flex-shrink-0" />
                    <span>{rec}</span>
                  </li>
                ))}
              </ul>
            </div>
          </div>

          {/* Action Buttons */}
          <div className="flex flex-col sm:flex-row items-center justify-end gap-4 mt-6">
            <button
              onClick={() => navigate('/upload')}
              className="w-full sm:w-auto px-6 py-3 border border-gray-200 text-gray-700 rounded-lg font-medium hover:bg-gray-50 transition-colors"
            >
              Analyze Another
            </button>
            <button
              onClick={() => navigate('/cam-report', { state: { loanId: effectiveLoanId } })}
              className="w-full sm:w-auto px-6 py-3 bg-primary text-white rounded-lg font-medium hover:bg-primary-light transition-colors flex items-center justify-center gap-2"
            >
              <Download className="w-4 h-4" />
              Download CAM Report
            </button>
          </div>
        </div>
      </main>
    </div>
  );
};

export default Results;
