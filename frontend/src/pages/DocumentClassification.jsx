import { useState, useEffect } from 'react';
import { useNavigate, useLocation } from 'react-router-dom';
import { 
  ArrowLeft, 
  ArrowRight, 
  Brain, 
  CheckCircle2, 
  XCircle, 
  RefreshCw,
  FileText,
  Scale,
  Users,
  Landmark,
  PieChart,
  AlertCircle,
  CheckCircle,
  ChevronDown
} from 'lucide-react';
import Navbar from '../components/Navbar';
import Sidebar from '../components/Sidebar';
import { documentTypes } from '../services/api';

const DocumentClassification = () => {
  const navigate = useNavigate();
  const location = useLocation();
  const { entityData, files } = location.state || {};
  
  const [sidebarOpen, setSidebarOpen] = useState(false);
  const [classifications, setClassifications] = useState([]);
  const [isProcessing, setIsProcessing] = useState(true);
  const [editingId, setEditingId] = useState(null);

  useEffect(() => {
    // Redirect if no data
    if (!entityData || !files) {
      navigate('/onboarding');
      return;
    }
    // Process file classifications based on uploaded files
    const timer = setTimeout(() => {
      // Create classifications from uploaded files
      const mappedClassifications = files.map((file, index) => {
        // Determine document type based on filename or use generic
        const fileName = file.file.name.toLowerCase();
        let detectedType = 'unknown';
        let confidence = 75;
        
        if (fileName.includes('annual') || fileName.includes('report')) {
          detectedType = 'annual_report';
          confidence = 92;
        } else if (fileName.includes('alm') || fileName.includes('asset')) {
          detectedType = 'alm_statement';
          confidence = 88;
        } else if (fileName.includes('shareholding') || fileName.includes('share')) {
          detectedType = 'shareholding_pattern';
          confidence = 90;
        } else if (fileName.includes('borrow') || fileName.includes('debt')) {
          detectedType = 'borrowing_profile';
          confidence = 85;
        } else if (fileName.includes('cash') || fileName.includes('flow')) {
          detectedType = 'cash_flow';
          confidence = 87;
        } else if (fileName.includes('portfolio') || fileName.includes('performance')) {
          detectedType = 'portfolio_performance';
          confidence = 83;
        } else if (fileName.includes('balance') || fileName.includes('sheet')) {
          detectedType = 'balance_sheet';
          confidence = 91;
        } else if (fileName.includes('pnl') || fileName.includes('profit') || fileName.includes('loss')) {
          detectedType = 'pnl_statement';
          confidence = 89;
        }
        
        return {
          id: file.id || index + 1,
          fileName: file.file.name,
          fileSize: file.file.size,
          uploadedType: file.type,
          detectedType: detectedType,
          confidence: confidence,
          status: 'pending'
        };
      });
      setClassifications(mappedClassifications);
      setIsProcessing(false);
    }, 2000);

    return () => clearTimeout(timer);
  }, [files]);

  const handleApprove = (id) => {
    setClassifications(prev => 
      prev.map(item => 
        item.id === id ? { ...item, status: 'approved' } : item
      )
    );
  };

  const handleReject = (id) => {
    setClassifications(prev => 
      prev.map(item => 
        item.id === id ? { ...item, status: 'rejected' } : item
      )
    );
  };

  const handleTypeChange = (id, newType) => {
    setClassifications(prev => 
      prev.map(item => 
        item.id === id ? { ...item, detectedType: newType, status: 'modified' } : item
      )
    );
    setEditingId(null);
  };

  const getStatusBadge = (status) => {
    switch (status) {
      case 'approved':
        return { bg: 'bg-accent/10', text: 'text-accent', icon: CheckCircle, label: 'Approved' };
      case 'rejected':
        return { bg: 'bg-danger/10', text: 'text-danger', icon: XCircle, label: 'Rejected' };
      case 'modified':
        return { bg: 'bg-warning/10', text: 'text-warning', icon: RefreshCw, label: 'Modified' };
      default:
        return { bg: 'bg-gray-100', text: 'text-gray-600', icon: AlertCircle, label: 'Pending' };
    }
  };

  const getDocumentIcon = (typeId) => {
    switch (typeId) {
      case 'alm': return Scale;
      case 'shareholding': return Users;
      case 'borrowing': return Landmark;
      case 'annual_report': return FileText;
      case 'portfolio': return PieChart;
      default: return FileText;
    }
  };

  const getConfidenceColor = (confidence) => {
    if (confidence >= 90) return 'text-accent';
    if (confidence >= 75) return 'text-warning';
    return 'text-danger';
  };

  const approvedCount = classifications.filter(c => c.status === 'approved' || c.status === 'modified').length;
  const totalCount = classifications.length;
  const canProceed = approvedCount > 0;

  const handleProceed = () => {
    const approvedDocs = classifications.filter(c => c.status === 'approved' || c.status === 'modified');
    navigate('/schema-mapping', { 
      state: { 
        entityData, 
        files, 
        classifications: approvedDocs 
      } 
    });
  };

  return (
    <div className="min-h-screen bg-background">
      <Navbar onMenuClick={() => setSidebarOpen(true)} />
      <Sidebar isOpen={sidebarOpen} onClose={() => setSidebarOpen(false)} />
      
      <main className="lg:ml-64 pt-16 min-h-screen">
        <div className="p-6 max-w-6xl mx-auto">
          {/* Header */}
          <div className="flex items-center gap-4 mb-8">
            <button 
              onClick={() => navigate('/upload', { state: { entityData } })}
              className="p-2 hover:bg-gray-100 rounded-lg transition-colors"
            >
              <ArrowLeft className="w-5 h-5 text-gray-600" />
            </button>
            <div>
              <h1 className="text-2xl font-bold text-primary">AI Document Classification</h1>
              <p className="text-gray-500 mt-1">Review and verify AI-detected document types</p>
            </div>
          </div>

          {/* Processing State */}
          {isProcessing ? (
            <div className="bg-white rounded-xl border border-gray-100 shadow-sm p-12 text-center">
              <div className="inline-flex items-center justify-center w-20 h-20 bg-primary/10 rounded-2xl mb-6">
                <Brain className="w-10 h-10 text-primary animate-pulse" />
              </div>
              <h2 className="text-xl font-semibold text-primary mb-2">AI Analyzing Documents...</h2>
              <p className="text-gray-500 mb-6">Our intelligent system is classifying your uploaded documents</p>
              <div className="max-w-md mx-auto">
                <div className="h-2 bg-gray-200 rounded-full overflow-hidden">
                  <div className="h-full bg-gradient-to-r from-primary to-secondary animate-pulse rounded-full" 
                       style={{ width: '60%' }} />
                </div>
              </div>
              <div className="mt-6 space-y-2">
                <p className="text-sm text-gray-400">Extracting text using OCR...</p>
                <p className="text-sm text-gray-400">Identifying document patterns...</p>
                <p className="text-sm text-gray-400">Matching with document templates...</p>
              </div>
            </div>
          ) : (
            <>
              {/* Progress Summary */}
              <div className="bg-white rounded-xl border border-gray-100 shadow-sm p-6 mb-6">
                <div className="flex items-center justify-between">
                  <div className="flex items-center gap-4">
                    <div className="w-12 h-12 bg-primary/10 rounded-xl flex items-center justify-center">
                      <CheckCircle2 className="w-6 h-6 text-primary" />
                    </div>
                    <div>
                      <h3 className="font-semibold text-primary">Classification Complete</h3>
                      <p className="text-sm text-gray-500">
                        {approvedCount} of {totalCount} documents approved
                      </p>
                    </div>
                  </div>
                  <div className="text-right">
                    <p className="text-2xl font-bold text-primary">{Math.round((approvedCount / totalCount) * 100)}%</p>
                    <p className="text-sm text-gray-500">Approved</p>
                  </div>
                </div>
                <div className="mt-4 h-2 bg-gray-100 rounded-full overflow-hidden">
                  <div 
                    className="h-full bg-accent rounded-full transition-all duration-500"
                    style={{ width: `${(approvedCount / totalCount) * 100}%` }}
                  />
                </div>
              </div>

              {/* Classification Table */}
              <div className="bg-white rounded-xl border border-gray-100 shadow-sm overflow-hidden">
                <div className="p-6 border-b border-gray-100">
                  <h2 className="text-lg font-semibold text-primary">Document Classification Results</h2>
                  <p className="text-sm text-gray-500 mt-1">
                    Review AI classifications. Approve, reject, or modify as needed.
                  </p>
                </div>

                <div className="overflow-x-auto">
                  <table className="w-full">
                    <thead className="bg-gray-50">
                      <tr>
                        <th className="px-6 py-4 text-left text-xs font-semibold text-gray-500 uppercase tracking-wider">File Name</th>
                        <th className="px-6 py-4 text-left text-xs font-semibold text-gray-500 uppercase tracking-wider">Detected Type</th>
                        <th className="px-6 py-4 text-left text-xs font-semibold text-gray-500 uppercase tracking-wider">Confidence</th>
                        <th className="px-6 py-4 text-left text-xs font-semibold text-gray-500 uppercase tracking-wider">Status</th>
                        <th className="px-6 py-4 text-left text-xs font-semibold text-gray-500 uppercase tracking-wider">Actions</th>
                      </tr>
                    </thead>
                    <tbody className="divide-y divide-gray-100">
                      {classifications.map((item) => {
                        const statusBadge = getStatusBadge(item.status);
                        const StatusIcon = statusBadge.icon;
                        const DocIcon = getDocumentIcon(item.detectedType);
                        const docType = documentTypes.find(t => t.id === item.detectedType);

                        return (
                          <tr key={item.id} className="hover:bg-gray-50 transition-colors">
                            <td className="px-6 py-4">
                              <div className="flex items-center gap-3">
                                <div className="w-10 h-10 bg-primary/10 rounded-lg flex items-center justify-center">
                                  <DocIcon className="w-5 h-5 text-primary" />
                                </div>
                                <div>
                                  <p className="font-medium text-gray-900">{item.fileName}</p>
                                  <p className="text-xs text-gray-500">
                                    {(item.fileSize / 1024 / 1024).toFixed(2)} MB
                                  </p>
                                </div>
                              </div>
                            </td>
                            <td className="px-6 py-4">
                              {editingId === item.id ? (
                                <select
                                  value={item.detectedType}
                                  onChange={(e) => handleTypeChange(item.id, e.target.value)}
                                  className="input-field text-sm py-1"
                                  autoFocus
                                >
                                  {documentTypes.map(type => (
                                    <option key={type.id} value={type.id}>{type.label}</option>
                                  ))}
                                </select>
                              ) : (
                                <div>
                                  <p className="font-medium text-gray-900">{docType?.label}</p>
                                  <button
                                    onClick={() => setEditingId(item.id)}
                                    className="text-xs text-primary hover:underline"
                                  >
                                    Change Type
                                  </button>
                                </div>
                              )}
                            </td>
                            <td className="px-6 py-4">
                              <div className="flex items-center gap-2">
                                <div className="w-16 h-2 bg-gray-200 rounded-full overflow-hidden">
                                  <div 
                                    className={`h-full rounded-full ${
                                      item.confidence >= 90 ? 'bg-accent' : 
                                      item.confidence >= 75 ? 'bg-warning' : 'bg-danger'
                                    }`}
                                    style={{ width: `${item.confidence}%` }}
                                  />
                                </div>
                                <span className={`text-sm font-medium ${getConfidenceColor(item.confidence)}`}>
                                  {item.confidence}%
                                </span>
                              </div>
                            </td>
                            <td className="px-6 py-4">
                              <span className={`inline-flex items-center gap-1.5 px-3 py-1 rounded-full text-xs font-medium ${statusBadge.bg} ${statusBadge.text}`}>
                                <StatusIcon className="w-3.5 h-3.5" />
                                {statusBadge.label}
                              </span>
                            </td>
                            <td className="px-6 py-4">
                              <div className="flex items-center gap-2">
                                {item.status !== 'approved' && item.status !== 'modified' && (
                                  <button
                                    onClick={() => handleApprove(item.id)}
                                    className="p-2 text-accent hover:bg-accent/10 rounded-lg transition-colors"
                                    title="Approve"
                                  >
                                    <CheckCircle className="w-5 h-5" />
                                  </button>
                                )}
                                {item.status !== 'rejected' && (
                                  <button
                                    onClick={() => handleReject(item.id)}
                                    className="p-2 text-danger hover:bg-danger/10 rounded-lg transition-colors"
                                    title="Reject"
                                  >
                                    <XCircle className="w-5 h-5" />
                                  </button>
                                )}
                              </div>
                            </td>
                          </tr>
                        );
                      })}
                    </tbody>
                  </table>
                </div>
              </div>

              {/* Legend */}
              <div className="mt-6 grid grid-cols-2 md:grid-cols-4 gap-4">
                {documentTypes.map((type) => {
                  const Icon = getDocumentIcon(type.id);
                  return (
                    <div key={type.id} className="flex items-center gap-2 text-sm text-gray-600">
                      <Icon className="w-4 h-4 text-gray-400" />
                      <span>{type.label}</span>
                    </div>
                  );
                })}
              </div>

              {/* Action Buttons */}
              <div className="flex items-center justify-between mt-8">
                <button
                  onClick={() => navigate('/upload', { state: { entityData } })}
                  className="flex items-center gap-2 px-6 py-3 border border-gray-200 text-gray-700 rounded-lg font-medium hover:bg-gray-50 transition-colors"
                >
                  <ArrowLeft className="w-4 h-4" />
                  Back to Upload
                </button>
                <button
                  onClick={handleProceed}
                  disabled={!canProceed}
                  className="flex items-center gap-2 px-8 py-3 bg-primary text-white rounded-lg font-medium hover:bg-primary-light transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
                >
                  Continue to Schema Mapping
                  <ArrowRight className="w-4 h-4" />
                </button>
              </div>
            </>
          )}
        </div>
      </main>
    </div>
  );
};

export default DocumentClassification;
