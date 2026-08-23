import { useState, useEffect } from 'react';
import { useNavigate, useLocation } from 'react-router-dom';
import { 
  FileSearch, 
  Database, 
  Scale, 
  Building2, 
  Brain, 
  FileText,
  CheckCircle2,
  Loader2,
  AlertCircle
} from 'lucide-react';
import { triggerAnalysis, getAnalysisStatus, createLoan } from '../services/api';

const processingSteps = [
  { id: 1, icon: FileSearch, text: 'Analyzing financial statements...', duration: 1500, category: 'document' },
  { id: 2, icon: Database, text: 'Extracting financial entities...', duration: 1500, category: 'extraction' },
  { id: 3, icon: Scale, text: 'Detecting circular trading patterns...', duration: 2000, category: 'risk' },
  { id: 4, icon: Building2, text: 'Scanning legal disputes...', duration: 1500, category: 'risk' },
  { id: 5, icon: FileSearch, text: 'Checking MCA filings...', duration: 1500, category: 'compliance' },
  { id: 6, icon: Brain, text: 'Analyzing market sentiment...', duration: 2000, category: 'research' },
  { id: 7, icon: Database, text: 'Evaluating promoter reputation...', duration: 1500, category: 'research' },
  { id: 8, icon: Brain, text: 'Running credit risk model...', duration: 2500, category: 'analysis' },
  { id: 9, icon: FileText, text: 'Generating CAM report...', duration: 2000, category: 'output' },
];

const financialFacts = [
  '40% of corporate defaults occur due to cashflow mismatch.',
  'GST mismatches often indicate potential revenue inflation.',
  'Circular trading patterns can be detected through network analysis.',
  'Companies with litigation history have 3x higher default rates.',
  'Sector concentration risk is a leading indicator of portfolio stress.',
  'Promoter credibility accounts for 15% of the risk score.',
  'ALM statements reveal liquidity stress 6 months before default.',
  'Shareholding concentration above 70% increases governance risk.',
  'Debt-Equity ratio above 2:1 signals high financial leverage.',
  'Positive news sentiment correlates with 20% lower default probability.',
];

const Processing = () => {
  const navigate = useNavigate();
  const location = useLocation();
  const [currentStep, setCurrentStep] = useState(0);
  const [progress, setProgress] = useState(0);
  const [factIndex, setFactIndex] = useState(0);
  const [completedSteps, setCompletedSteps] = useState([]);
  const [error, setError] = useState(null);
  const [loanId, setLoanId] = useState(null);
  const [analysisState, setAnalysisState] = useState('idle'); // idle, creating, triggering, polling, completed, failed
  const [statusMessage, setStatusMessage] = useState('Initializing...');
  const [pollCount, setPollCount] = useState(0);

  useEffect(() => {
    if (!location.state) {
      navigate('/upload');
      return;
    }

    const { entityData } = location.state;
    
    // Validate entity data has required id
    if (!entityData || !entityData.id) {
      console.error('Missing entityData.id:', entityData);
      setError('Invalid entity data. Please start from onboarding.');
      return;
    }
    
    let pollInterval = null;
    let visualInterval = null;
    
    // Start real analysis process
    const startAnalysis = async () => {
      try {
        setError(null);
        setAnalysisState('creating');
        setStatusMessage('Creating loan application...');
        setCurrentStep(0);
        
        // Step 1: Create loan application
        const loanData = {
          entity_id: entityData.id,
          loan_type: 'Term Loan',
          loan_amount: entityData.loan_amount || entityData.loanAmount || 10.0,
          tenure_months: 60,
          loan_purpose: 'Working Capital'
        };
        
        console.log('Creating loan with data:', loanData);
        const loan = await createLoan(entityData.id, loanData);
        if (!loan || !loan.id) {
          throw new Error('Failed to create loan: No loan ID returned from server');
        }
        const currentLoanId = loan.id;
        setLoanId(currentLoanId);
        localStorage.setItem('last_loan_id', currentLoanId);
        console.log('Loan created:', currentLoanId);
        
        // Step 2: Trigger analysis
        setAnalysisState('triggering');
        setStatusMessage('Starting AI analysis...');
        setCurrentStep(1);
        console.log('Triggering analysis for loan:', currentLoanId);
        await triggerAnalysis(currentLoanId);
        
        // Step 3: Poll for completion
        setAnalysisState('polling');
        setStatusMessage('Analyzing documents...');
        let localPollCount = 0;
        
        pollInterval = setInterval(async () => {
          try {
            localPollCount++;
            setPollCount(localPollCount);
            console.log(`Polling attempt ${localPollCount}...`);
            
            const status = await getAnalysisStatus(currentLoanId);
            console.log('Analysis status:', status);
            
            // Update progress based on status
            const statusStr = status.status || '';
            
            if (statusStr === 'pending') {
              setProgress(prev => Math.max(prev, 20));
              setCurrentStep(2);
              setStatusMessage('Waiting to start analysis...');
            } else if (statusStr === 'running' || statusStr.startsWith('running:')) {
              // Handle granular status updates from backend
              if (statusStr === 'running:extracting') {
                setProgress(30);
                setCurrentStep(1);
                setStatusMessage('Extracting financial statements...');
              } else if (statusStr === 'running:scoring') {
                setProgress(60);
                setCurrentStep(7);
                setStatusMessage('Calculating risk score...');
              } else if (statusStr === 'running:training_ml') {
                setProgress(68);
                setCurrentStep(7);
                setStatusMessage('Training ML models on historical data...');
              } else if (statusStr === 'running:rescoring') {
                setProgress(75);
                setCurrentStep(7);
                setStatusMessage('Re-scoring with ensemble models...');
              } else if (statusStr === 'running:scoring_complete') {
                setProgress(82);
                setCurrentStep(7);
                setStatusMessage('Risk score calculated. Preparing SWOT analysis...');
              } else if (statusStr === 'running:swot') {
                setProgress(85);
                setCurrentStep(7);
                setStatusMessage('Generating SWOT analysis...');
              } else if (statusStr === 'running:cam') {
                setProgress(92);
                setCurrentStep(8);
                setStatusMessage('Generating CAM report...');
              } else if (statusStr === 'running:pdf') {
                setProgress(96);
                setCurrentStep(8);
                setStatusMessage('Exporting PDF report...');
              } else {
                // General running status - gradually increase progress
                const baseProgress = 25;
                const maxRunningProgress = 85;
                const incrementPerPoll = (maxRunningProgress - baseProgress) / 50;
                const calculatedProgress = baseProgress + (localPollCount * incrementPerPoll);
                const newProgress = Math.min(calculatedProgress, maxRunningProgress);
                
                setProgress(prev => Math.max(prev, newProgress));
                
                const stepIndex = Math.min(7, 3 + Math.floor(localPollCount / 4));
                setCurrentStep(stepIndex);
                
                // Update status message based on progress
                const messages = [
                  'Extracting financial data...',
                  'Extracting financial entities...',
                  'Computing financial ratios...',
                  'Detecting circular trading patterns...',
                  'Scanning legal disputes...',
                  'Checking MCA filings...',
                  'Analyzing market sentiment...',
                  'Evaluating promoter reputation...',
                  'Calculating risk score...'
                ];
                const messageIndex = Math.min(Math.floor(localPollCount / 3), messages.length - 1);
                setStatusMessage(messages[messageIndex]);
              }
            } else if (statusStr === 'running:scoring_complete') {
              // Risk score is ready even if later steps (SWOT/report generation) are still running.
              // Navigate early so the user sees results without waiting for the full report.
              setProgress(85);
              setCurrentStep(7);
              setStatusMessage('Risk score ready - displaying results...');
              setAnalysisState('completed');
              clearInterval(pollInterval);
              setTimeout(() => {
                if (currentLoanId) {
                  navigate('/results', { state: { ...location.state, loanId: currentLoanId } });
                } else {
                  setError('Analysis completed but loan ID is missing. Please try again.');
                }
              }, 800);
            } else if (statusStr === 'completed') {
              setProgress(100);
              setCurrentStep(processingSteps.length - 1);
              setCompletedSteps(processingSteps.map((_, i) => i));
              setAnalysisState('completed');
              setStatusMessage('Analysis complete!');
              clearInterval(pollInterval);
              setTimeout(() => {
                if (currentLoanId) {
                  navigate('/results', { state: { ...location.state, loanId: currentLoanId } });
                } else {
                  setError('Analysis completed but loan ID is missing. Please try again.');
                }
              }, 1500);
            } else if (statusStr === 'failed') {
              clearInterval(pollInterval);
              setAnalysisState('failed');
              setStatusMessage('Analysis failed');
              // Get error message from backend if available
              const errorMsg = status.error || status.message || 'Unknown error';
              if (errorMsg.toLowerCase().includes('no financial data')) {
                setError(
                  'Analysis failed: No financial data could be extracted from the uploaded documents. ' +
                  'Please upload readable PDFs or well-structured Excel/CSV files with financial statements and try again.'
                );
              } else {
                setError(`Analysis failed: ${errorMsg}. Please check if Ollama is running and try again.`);
              }
            }
            
            // Timeout after 5 minutes (100 polls * 3 seconds)
            if (localPollCount > 100) {
              clearInterval(pollInterval);
              setAnalysisState('failed');
              setError('Analysis timed out. The backend may be stuck or Ollama may not be responding.');
            }
          } catch (err) {
            console.error('Polling error:', err);
            setStatusMessage(`Checking status... (attempt ${localPollCount})`);
            // Don't stop polling on error, keep trying
          }
        }, 3000); // Poll every 3 seconds
        
      } catch (err) {
        console.error('Analysis error:', err);
        setAnalysisState('failed');
        setError('Failed to start analysis: ' + (err.message || 'Unknown error'));
      }
    };
    
    // Visual simulation is now integrated into the polling logic
    // No separate visual interval needed - progress updates come from real backend status
    
    // Start real analysis
    startAnalysis();

    // Cleanup function
    return () => {
      if (pollInterval) clearInterval(pollInterval);
    };
  }, []); // Empty dependency array - run once on mount

  // Rotate facts
  useEffect(() => {
    const factInterval = setInterval(() => {
      setFactIndex((prev) => (prev + 1) % financialFacts.length);
    }, 4000);

    return () => clearInterval(factInterval);
  }, []);

  return (
    <div className="min-h-screen bg-background flex items-center justify-center p-4">
      <div className="w-full max-w-2xl">
        {/* Header */}
        <div className="text-center mb-10">
          <div className="inline-flex items-center justify-center w-20 h-20 bg-primary/10 rounded-2xl mb-6">
            <div className="w-12 h-12 bg-primary rounded-xl flex items-center justify-center">
              <Brain className="w-7 h-7 text-white animate-pulse" />
            </div>
          </div>
          <h1 className="text-3xl font-bold text-primary mb-2">
            {error ? 'Analysis Error' : 'AI Analysis in Progress'}
          </h1>
          <p className="text-gray-500">
            {error ? 'Something went wrong during analysis' : statusMessage}
          </p>
          {analysisState === 'polling' && (
            <p className="text-xs text-gray-400 mt-2">
              Poll count: {pollCount} | Loan ID: {loanId?.substring(0, 8)}...
            </p>
          )}
        </div>
        
        {/* Error Display */}
        {error && (
          <div className="bg-danger/10 border border-danger/20 rounded-xl p-6 mb-6">
            <div className="flex items-center gap-3">
              <AlertCircle className="w-6 h-6 text-danger" />
              <div>
                <h3 className="font-semibold text-danger">Error</h3>
                <p className="text-sm text-gray-600">{error}</p>
              </div>
            </div>
            <div className="flex gap-3 mt-4">
              <button
                onClick={() => navigate('/upload')}
                className="px-4 py-2 bg-primary text-white rounded-lg text-sm font-medium hover:bg-primary-light transition-colors"
              >
                Try Again
              </button>
              {loanId && (
                <button
                  onClick={() => navigate('/results', { state: { ...location.state, loanId } })}
                  className="px-4 py-2 border border-gray-300 text-gray-700 rounded-lg text-sm font-medium hover:bg-gray-50 transition-colors"
                >
                  Skip to Results (Demo)
                </button>
              )}
            </div>
          </div>
        )}
        
        {/* Timeout Warning */}
        {analysisState === 'polling' && pollCount > 30 && (
          <div className="bg-warning/10 border border-warning/20 rounded-xl p-4 mb-6">
            <div className="flex items-center gap-3">
              <AlertCircle className="w-5 h-5 text-warning" />
              <div>
                <p className="text-sm text-gray-700">Analysis is taking longer than expected.</p>
                <p className="text-xs text-gray-500 mt-1">Make sure Ollama is running: ollama serve</p>
              </div>
            </div>
            <button
              onClick={() => navigate('/results', { state: { ...location.state, loanId } })}
              className="mt-3 px-4 py-2 bg-warning text-white rounded-lg text-sm font-medium hover:bg-warning-dark transition-colors"
            >
              Skip to Results (Use Demo Data)
            </button>
          </div>
        )}

        {/* Progress Bar */}
        <div className="mb-10">
          <div className="h-2 bg-gray-200 rounded-full overflow-hidden">
            <div 
              className="h-full bg-gradient-to-r from-primary to-secondary transition-all duration-300 ease-out"
              style={{ width: `${progress}%` }}
            />
          </div>
          <div className="flex justify-between mt-2 text-sm text-gray-500">
            <span>Processing</span>
            <span>{Math.round(progress)}%</span>
          </div>
        </div>

        {/* Processing Steps */}
        <div className="bg-white rounded-2xl shadow-sm border border-gray-100 p-6 mb-8">
          <div className="space-y-4">
            {processingSteps.map((step, index) => {
              const Icon = step.icon;
              const isCompleted = completedSteps.includes(index);
              const isCurrent = index === currentStep;
              const _isPending = index > currentStep;

              return (
                <div 
                  key={step.id}
                  className={`flex items-center gap-4 p-4 rounded-xl transition-all duration-300 ${
                    isCurrent ? 'bg-primary/5 border border-primary/20' : 
                    isCompleted ? 'bg-gray-50' : 'opacity-50'
                  }`}
                >
                  <div className={`w-10 h-10 rounded-lg flex items-center justify-center ${
                    isCompleted ? 'bg-accent' : 
                    isCurrent ? 'bg-primary' : 'bg-gray-200'
                  }`}>
                    {isCompleted ? (
                      <CheckCircle2 className="w-5 h-5 text-white" />
                    ) : isCurrent ? (
                      <Loader2 className="w-5 h-5 text-white animate-spin" />
                    ) : (
                      <Icon className="w-5 h-5 text-gray-400" />
                    )}
                  </div>
                  <div className="flex-1">
                    <p className={`font-medium ${
                      isCurrent ? 'text-primary' : 
                      isCompleted ? 'text-gray-700' : 'text-gray-400'
                    }`}>
                      {step.text}
                    </p>
                  </div>
                  {isCurrent && (
                    <div className="flex gap-1">
                      <span className="w-2 h-2 bg-primary rounded-full animate-bounce"></span>
                      <span className="w-2 h-2 bg-primary rounded-full animate-bounce" style={{ animationDelay: '0.1s' }}></span>
                      <span className="w-2 h-2 bg-primary rounded-full animate-bounce" style={{ animationDelay: '0.2s' }}></span>
                    </div>
                  )}
                </div>
              );
            })}
          </div>
        </div>

        {/* Financial Facts */}
        <div className="bg-gradient-to-br from-primary to-secondary rounded-2xl p-6 text-white text-center">
          <p className="text-white/60 text-sm mb-2">Did you know?</p>
          <p 
            key={factIndex}
            className="text-lg font-medium animate-fade-in"
          >
            {financialFacts[factIndex]}
          </p>
        </div>
      </div>
    </div>
  );
};

export default Processing;

