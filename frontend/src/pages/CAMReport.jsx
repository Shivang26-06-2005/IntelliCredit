import { useState, useEffect } from 'react';
import { useNavigate, useLocation } from 'react-router-dom';
import { 
  ArrowLeft, 
  Download, 
  Building2, 
  User, 
  Wallet, 
  PiggyBank, 
  Shield, 
  TrendingUp,
  FileCheck,
  AlertCircle,
  CheckCircle2,
  Target,
  Zap,
  AlertTriangle,
  Crosshair,
  Loader2
} from 'lucide-react';
import Navbar from '../components/Navbar';
import Sidebar from '../components/Sidebar';
import { getCAMReport, downloadReport } from '../services/api';

const CAMReport = () => {
  const navigate = useNavigate();
  const location = useLocation();
  const [sidebarOpen, setSidebarOpen] = useState(false);
  const [data, setData] = useState(null);
  const [loading, setLoading] = useState(true);
  const [isDownloadingCam, setIsDownloadingCam] = useState(false);
  const [error, setError] = useState(null);
  const [loanId, setLoanId] = useState(null);

  useEffect(() => {
    const fetchData = async () => {
      try {
        setLoading(true);
        setError(null);
        
        // Get loanId from navigation state
        const resolvedLoanId = location.state?.loanId || localStorage.getItem('last_loan_id');
        
        if (!resolvedLoanId) {
          throw new Error('No loan ID found. Please complete the analysis first.');
        }
        localStorage.setItem('last_loan_id', resolvedLoanId);
        setLoanId(resolvedLoanId);
        
        const result = await getCAMReport(resolvedLoanId);
        
        // If we have real report content from backend, use it
        // Otherwise merge with mock SWOT data
        const enhancedResult = {
          ...result,
          swotAnalysis: result.swotAnalysis || {
            strengths: [
              'Consistent revenue growth of 15% YoY',
              'Strong market position in auto components',
              'Diversified client base across OEMs',
              'Experienced management team with 15+ years',
              'Strong GST compliance history'
            ],
            weaknesses: [
              'High debt-to-equity ratio of 1.8:1',
              'Factory utilization at only 45% capacity',
              'Dependence on top 3 clients (60% revenue)',
              'Limited geographic diversification'
            ],
            opportunities: [
              'Expansion into EV component manufacturing',
              'Government PLI scheme benefits',
              'Growing auto component sector at 12% CAGR',
              'Export opportunities to Southeast Asia'
            ],
            threats: [
              'Regulatory changes in emission norms',
              'Ongoing litigation cases (2 pending)',
              'Raw material price volatility',
              'Competition from low-cost imports'
            ]
          }
        };
        setData(enhancedResult);
      } catch (err) {
        console.error('Failed to fetch CAM report:', err);
        setError(err.message || 'Failed to load CAM report. Please try again.');
      } finally {
        setLoading(false);
      }
    };
    fetchData();
  }, [location.state]);

  const getScoreBadge = (score) => {
    if (score >= 80) return 'bg-accent/10 text-accent';
    if (score >= 60) return 'bg-warning/10 text-warning';
    return 'bg-danger/10 text-danger';
  };

  const handleDownloadCamPdf = async () => {
    if (!loanId) {
      setError('No loan ID available to download CAM PDF.');
      return;
    }
    try {
      setIsDownloadingCam(true);
      const blob = await downloadReport(loanId);
      const url = window.URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = `CAM_Report_${loanId.slice(0, 8)}.pdf`;
      document.body.appendChild(a);
      a.click();
      a.remove();
      window.URL.revokeObjectURL(url);
    } catch (err) {
      console.error('Failed to download CAM PDF:', err);
      setError(err.message || 'Failed to download CAM PDF.');
    } finally {
      setIsDownloadingCam(false);
    }
  };

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
            <h1 className="text-2xl font-bold text-primary mb-2">Error Loading CAM Report</h1>
            <p className="text-gray-500 mb-6">{error}</p>
            <button
              onClick={() => navigate('/results')}
              className="px-6 py-3 bg-primary text-white rounded-lg hover:bg-primary-light transition-colors"
            >
              Go Back to Results
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
        <div className="p-6 max-w-5xl mx-auto">
          {/* Header */}
          <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4 mb-8">
            <div className="flex items-center gap-4">
              <button 
                onClick={() => navigate('/results')}
                className="p-2 hover:bg-gray-100 rounded-lg transition-colors"
              >
                <ArrowLeft className="w-5 h-5 text-gray-600" />
              </button>
              <div>
                <h1 className="text-2xl font-bold text-primary">Credit Appraisal Memo</h1>
                <p className="text-gray-500 mt-1">Comprehensive credit assessment report</p>
              </div>
            </div>
            
            {/* Error Display */}
            {error && (
              <div className="bg-warning/10 border border-warning/20 rounded-xl p-4">
                <div className="flex items-center gap-3">
                  <AlertCircle className="w-5 h-5 text-warning" />
                  <p className="text-sm text-gray-600">{error}</p>
                </div>
              </div>
            )}
            <div className="flex items-center gap-3">
              <button
                onClick={handleDownloadCamPdf}
                disabled={isDownloadingCam}
                className="flex items-center justify-center gap-2 px-6 py-3 bg-primary text-white rounded-lg hover:bg-primary-light transition-colors disabled:opacity-50"
              >
                {isDownloadingCam ? (
                  <>
                    <Loader2 className="w-4 h-4 animate-spin" />
                    Downloading...
                  </>
                ) : (
                  <>
                    <Download className="w-4 h-4" />
                    Download CAM PDF
                  </>
                )}
              </button>
            </div>
          </div>

          {/* CAM Report Content */}
          <div className="bg-white rounded-xl border border-gray-100 shadow-sm overflow-hidden">
            {/* Report Header */}
            <div className="bg-gradient-to-r from-primary to-secondary p-8 text-white">
              <div className="flex items-center gap-4 mb-4">
                <div className="w-16 h-16 bg-white/20 rounded-xl flex items-center justify-center">
                  <FileCheck className="w-8 h-8" />
                </div>
                <div>
                  <h2 className="text-2xl font-bold">Credit Appraisal Memo</h2>
                  <p className="text-white/70">Generated by IntelliCredit AI</p>
                </div>
              </div>
              <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mt-6 pt-6 border-t border-white/20">
                <div>
                  <p className="text-white/60 text-sm">Company</p>
                  <p className="font-semibold">{data?.companyOverview?.name}</p>
                </div>
                <div>
                  <p className="text-white/60 text-sm">Industry</p>
                  <p className="font-semibold">{data?.companyOverview?.industry}</p>
                </div>
                <div>
                  <p className="text-white/60 text-sm">Report Date</p>
                  <p className="font-semibold">{new Date().toLocaleDateString('en-IN')}</p>
                </div>
                <div>
                  <p className="text-white/60 text-sm">Reference ID</p>
                  <p className="font-semibold">CAM-2024031201</p>
                </div>
              </div>
            </div>

            {/* Report Body */}
            <div className="p-8 space-y-8">
              {/* Company Overview */}
              <section>
                <div className="flex items-center gap-3 mb-4">
                  <div className="w-10 h-10 bg-primary/10 rounded-lg flex items-center justify-center">
                    <Building2 className="w-5 h-5 text-primary" />
                  </div>
                  <h3 className="text-xl font-semibold text-primary">1. Company Overview</h3>
                </div>
                <div className="bg-gray-50 rounded-lg p-6">
                  <div className="grid grid-cols-2 md:grid-cols-4 gap-6">
                    <div>
                      <p className="text-sm text-gray-500">Company Name</p>
                      <p className="font-medium text-gray-900">{data?.companyOverview?.name}</p>
                    </div>
                    <div>
                      <p className="text-sm text-gray-500">Industry</p>
                      <p className="font-medium text-gray-900">{data?.companyOverview?.industry}</p>
                    </div>
                    <div>
                      <p className="text-sm text-gray-500">Established</p>
                      <p className="font-medium text-gray-900">{data?.companyOverview?.established}</p>
                    </div>
                    <div>
                      <p className="text-sm text-gray-500">Employees</p>
                      <p className="font-medium text-gray-900">{data?.companyOverview?.employees}</p>
                    </div>
                  </div>
                  <div className="mt-4 pt-4 border-t border-gray-200">
                    <p className="text-sm text-gray-500">Annual Turnover</p>
                    <p className="text-2xl font-bold text-primary">{data?.companyOverview?.turnover}</p>
                  </div>
                </div>
              </section>

              {/* Five Cs of Credit */}
              <section>
                <h3 className="text-xl font-semibold text-primary mb-6">2. Five Cs of Credit Analysis</h3>
                
                <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                  {/* Character */}
                  <div className="border border-gray-100 rounded-xl p-6">
                    <div className="flex items-center justify-between mb-4">
                      <div className="flex items-center gap-3">
                        <div className="w-10 h-10 bg-blue-100 rounded-lg flex items-center justify-center">
                          <User className="w-5 h-5 text-blue-600" />
                        </div>
                        <div>
                          <h4 className="font-semibold text-primary">Character</h4>
                          <p className="text-sm text-gray-500">Promoter Credibility</p>
                        </div>
                      </div>
                      <span className={`px-3 py-1 rounded-full text-sm font-medium ${getScoreBadge(data?.character?.score)}`}>
                        Score: {data?.character?.score}/100
                      </span>
                    </div>
                    <p className="text-gray-600 text-sm">{data?.character?.details}</p>
                  </div>

                  {/* Capacity */}
                  <div className="border border-gray-100 rounded-xl p-6">
                    <div className="flex items-center justify-between mb-4">
                      <div className="flex items-center gap-3">
                        <div className="w-10 h-10 bg-green-100 rounded-lg flex items-center justify-center">
                          <Wallet className="w-5 h-5 text-green-600" />
                        </div>
                        <div>
                          <h4 className="font-semibold text-primary">Capacity</h4>
                          <p className="text-sm text-gray-500">Repayment Ability</p>
                        </div>
                      </div>
                      <span className={`px-3 py-1 rounded-full text-sm font-medium ${getScoreBadge(data?.capacity?.score)}`}>
                        Score: {data?.capacity?.score}/100
                      </span>
                    </div>
                    <p className="text-gray-600 text-sm">{data?.capacity?.details}</p>
                  </div>

                  {/* Capital */}
                  <div className="border border-gray-100 rounded-xl p-6">
                    <div className="flex items-center justify-between mb-4">
                      <div className="flex items-center gap-3">
                        <div className="w-10 h-10 bg-purple-100 rounded-lg flex items-center justify-center">
                          <PiggyBank className="w-5 h-5 text-purple-600" />
                        </div>
                        <div>
                          <h4 className="font-semibold text-primary">Capital</h4>
                          <p className="text-sm text-gray-500">Financial Strength</p>
                        </div>
                      </div>
                      <span className={`px-3 py-1 rounded-full text-sm font-medium ${getScoreBadge(data?.capital?.score)}`}>
                        Score: {data?.capital?.score}/100
                      </span>
                    </div>
                    <p className="text-gray-600 text-sm">{data?.capital?.details}</p>
                  </div>

                  {/* Collateral */}
                  <div className="border border-gray-100 rounded-xl p-6">
                    <div className="flex items-center justify-between mb-4">
                      <div className="flex items-center gap-3">
                        <div className="w-10 h-10 bg-orange-100 rounded-lg flex items-center justify-center">
                          <Shield className="w-5 h-5 text-orange-600" />
                        </div>
                        <div>
                          <h4 className="font-semibold text-primary">Collateral</h4>
                          <p className="text-sm text-gray-500">Assets Pledged</p>
                        </div>
                      </div>
                      <span className="px-3 py-1 rounded-full text-sm font-medium bg-gray-100 text-gray-700">
                        Coverage: {data?.collateral?.coverage}
                      </span>
                    </div>
                    <p className="text-gray-600 text-sm">{data?.collateral?.details}</p>
                  </div>
                </div>

                {/* Conditions */}
                <div className="border border-gray-100 rounded-xl p-6 mt-6">
                  <div className="flex items-center justify-between mb-4">
                    <div className="flex items-center gap-3">
                      <div className="w-10 h-10 bg-cyan-100 rounded-lg flex items-center justify-center">
                        <TrendingUp className="w-5 h-5 text-cyan-600" />
                      </div>
                      <div>
                        <h4 className="font-semibold text-primary">Conditions</h4>
                        <p className="text-sm text-gray-500">Industry Outlook</p>
                      </div>
                    </div>
                    <span className={`px-3 py-1 rounded-full text-sm font-medium ${getScoreBadge(data?.conditions?.score)}`}>
                      Score: {data?.conditions?.score}/100
                    </span>
                  </div>
                  <p className="text-gray-600 text-sm">{data?.conditions?.details}</p>
                </div>
              </section>

              {/* Risk Assessment */}
              <section>
                <div className="flex items-center gap-3 mb-4">
                  <div className="w-10 h-10 bg-danger/10 rounded-lg flex items-center justify-center">
                    <AlertCircle className="w-5 h-5 text-danger" />
                  </div>
                  <h3 className="text-xl font-semibold text-primary">3. Risk Assessment</h3>
                </div>
                
                <div className="bg-gray-50 rounded-lg p-6">
                  <div className="flex items-center gap-4 mb-6">
                    <div className="text-center">
                      <p className="text-3xl font-bold text-primary">{data?.riskAssessment?.overallScore}</p>
                      <p className="text-sm text-gray-500">Risk Score</p>
                    </div>
                    <div className="flex-1 h-px bg-gray-200"></div>
                    <div>
                      <span className="px-4 py-2 bg-warning/10 text-warning rounded-full font-medium">
                        {data?.riskAssessment?.riskCategory}
                      </span>
                    </div>
                  </div>

                  <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                    <div>
                      <h5 className="font-medium text-gray-900 mb-3">Key Risks</h5>
                      <ul className="space-y-2">
                        {data?.riskAssessment?.keyRisks?.map((risk, index) => (
                          <li key={index} className="flex items-start gap-2 text-sm text-gray-600">
                            <span className="w-1.5 h-1.5 bg-danger rounded-full mt-1.5 flex-shrink-0"></span>
                            {risk}
                          </li>
                        ))}
                      </ul>
                    </div>
                    <div>
                      <h5 className="font-medium text-gray-900 mb-3">Mitigants</h5>
                      <ul className="space-y-2">
                        {data?.riskAssessment?.mitigants?.map((item, index) => (
                          <li key={index} className="flex items-start gap-2 text-sm text-gray-600">
                            <CheckCircle2 className="w-4 h-4 text-accent mt-0.5 flex-shrink-0" />
                            {item}
                          </li>
                        ))}
                      </ul>
                    </div>
                  </div>
                </div>
              </section>

              {/* SWOT Analysis */}
              <section>
                <div className="flex items-center gap-3 mb-6">
                  <div className="w-10 h-10 bg-primary/10 rounded-lg flex items-center justify-center">
                    <Target className="w-5 h-5 text-primary" />
                  </div>
                  <h3 className="text-xl font-semibold text-primary">4. SWOT Analysis</h3>
                </div>

                <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                  {/* Strengths */}
                  <div className="bg-green-50 rounded-xl p-5 border border-green-100">
                    <div className="flex items-center gap-2 mb-4">
                      <div className="w-8 h-8 bg-green-100 rounded-lg flex items-center justify-center">
                        <Zap className="w-4 h-4 text-green-600" />
                      </div>
                      <h4 className="font-semibold text-green-800">Strengths</h4>
                    </div>
                    <ul className="space-y-2">
                      {data?.swotAnalysis?.strengths?.map((item, index) => (
                        <li key={index} className="flex items-start gap-2 text-sm text-green-700">
                          <CheckCircle2 className="w-4 h-4 mt-0.5 flex-shrink-0" />
                          <span>{item}</span>
                        </li>
                      ))}
                    </ul>
                  </div>

                  {/* Weaknesses */}
                  <div className="bg-red-50 rounded-xl p-5 border border-red-100">
                    <div className="flex items-center gap-2 mb-4">
                      <div className="w-8 h-8 bg-red-100 rounded-lg flex items-center justify-center">
                        <AlertTriangle className="w-4 h-4 text-red-600" />
                      </div>
                      <h4 className="font-semibold text-red-800">Weaknesses</h4>
                    </div>
                    <ul className="space-y-2">
                      {data?.swotAnalysis?.weaknesses?.map((item, index) => (
                        <li key={index} className="flex items-start gap-2 text-sm text-red-700">
                          <AlertCircle className="w-4 h-4 mt-0.5 flex-shrink-0" />
                          <span>{item}</span>
                        </li>
                      ))}
                    </ul>
                  </div>

                  {/* Opportunities */}
                  <div className="bg-blue-50 rounded-xl p-5 border border-blue-100">
                    <div className="flex items-center gap-2 mb-4">
                      <div className="w-8 h-8 bg-blue-100 rounded-lg flex items-center justify-center">
                        <TrendingUp className="w-4 h-4 text-blue-600" />
                      </div>
                      <h4 className="font-semibold text-blue-800">Opportunities</h4>
                    </div>
                    <ul className="space-y-2">
                      {data?.swotAnalysis?.opportunities?.map((item, index) => (
                        <li key={index} className="flex items-start gap-2 text-sm text-blue-700">
                          <TrendingUp className="w-4 h-4 mt-0.5 flex-shrink-0" />
                          <span>{item}</span>
                        </li>
                      ))}
                    </ul>
                  </div>

                  {/* Threats */}
                  <div className="bg-orange-50 rounded-xl p-5 border border-orange-100">
                    <div className="flex items-center gap-2 mb-4">
                      <div className="w-8 h-8 bg-orange-100 rounded-lg flex items-center justify-center">
                        <Crosshair className="w-4 h-4 text-orange-600" />
                      </div>
                      <h4 className="font-semibold text-orange-800">Threats</h4>
                    </div>
                    <ul className="space-y-2">
                      {data?.swotAnalysis?.threats?.map((item, index) => (
                        <li key={index} className="flex items-start gap-2 text-sm text-orange-700">
                          <AlertTriangle className="w-4 h-4 mt-0.5 flex-shrink-0" />
                          <span>{item}</span>
                        </li>
                      ))}
                    </ul>
                  </div>
                </div>
              </section>

              {/* Final Recommendation */}
              <section>
                <div className="flex items-center gap-3 mb-4">
                  <div className="w-10 h-10 bg-accent/10 rounded-lg flex items-center justify-center">
                    <FileCheck className="w-5 h-5 text-accent" />
                  </div>
                  <h3 className="text-xl font-semibold text-primary">5. Final Recommendation</h3>
                </div>
                
                <div className="bg-gradient-to-r from-primary to-secondary rounded-xl p-6 text-white">
                  <div className="flex flex-col md:flex-row md:items-center justify-between gap-6">
                    <div>
                      <p className="text-white/70 text-sm mb-1">Decision</p>
                      <p className="text-2xl font-bold">{data?.recommendation?.decision}</p>
                    </div>
                    <div className="md:text-center">
                      <p className="text-white/70 text-sm mb-1">Loan Amount</p>
                      <p className="text-2xl font-bold">{data?.recommendation?.loanAmount}</p>
                    </div>
                    <div className="md:text-right">
                      <p className="text-white/70 text-sm mb-1">Interest Rate</p>
                      <p className="text-2xl font-bold">{data?.recommendation?.interestRate}</p>
                    </div>
                  </div>
                  
                  <div className="mt-6 pt-6 border-t border-white/20">
                    <p className="text-white/70 text-sm mb-2">Conditions</p>
                    <ul className="space-y-1">
                      {data?.recommendation?.conditions?.map((condition, index) => (
                        <li key={index} className="text-sm text-white/90">
                          {index + 1}. {condition}
                        </li>
                      ))}
                    </ul>
                  </div>
                </div>
              </section>

              {/* Footer */}
              <div className="pt-8 border-t border-gray-100 text-center">
                <p className="text-sm text-gray-500">
                  This Credit Appraisal Memo is generated by IntelliCredit AI and should be reviewed by a qualified credit officer before final approval.
                </p>
                <p className="text-xs text-gray-400 mt-2">
                  Report ID: CAM-2024031201 | Generated on: {new Date().toLocaleDateString('en-IN')}
                </p>
              </div>
            </div>
          </div>
        </div>
      </main>
    </div>
  );
};

export default CAMReport;
