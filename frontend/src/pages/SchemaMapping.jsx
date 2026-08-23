import { useState, useEffect } from 'react';
import { useNavigate, useLocation } from 'react-router-dom';
import { 
  ArrowLeft, 
  ArrowRight, 
  Database, 
  CheckCircle2, 
  AlertCircle,
  RefreshCw,
  Link2,
  Unlink,
  ChevronDown,
  Brain,
  FileText,
  Settings2
} from 'lucide-react';
import Navbar from '../components/Navbar';
import Sidebar from '../components/Sidebar';
import { schemaFields, listDocuments, getEntity } from '../services/api';

const SchemaMapping = () => {
  const navigate = useNavigate();
  const location = useLocation();
  const { entityData, files, classifications } = location.state || {};
  
  const [sidebarOpen, setSidebarOpen] = useState(false);
  const [mappings, setMappings] = useState([]);
  const [isProcessing, setIsProcessing] = useState(true);
  const [editingMapping, setEditingMapping] = useState(null);
  const [confirmedCount, setConfirmedCount] = useState(0);
  const [error, setError] = useState(null);

  useEffect(() => {
    // Redirect if no data
    if (!entityData || !classifications) {
      navigate('/onboarding');
      return;
    }
    
    // Fetch real schema data from backend
    const fetchSchemaData = async () => {
      try {
        setIsProcessing(true);
        setError(null);
        
        // Get documents for this entity
        const documents = await listDocuments(entityData.id);
        
        // Create schema mappings from document metadata
        let extractedFields = [];
        if (documents && documents.length > 0) {
          // Create mappings from document classifications
          extractedFields = documents.map((doc, index) => ({
            id: index + 1,
            extractedField: doc.document_type || 'Unknown Field',
            detectedValue: doc.filename || 'N/A',
            mappedTo: doc.document_type || 'unknown',
            confidence: 85,
            status: 'auto-mapped',
            source: 'backend'
          }));
        } else {
          // No documents - show empty state
          extractedFields = [];
        }
        
        setMappings(extractedFields);
        setConfirmedCount(extractedFields.length);
        setIsProcessing(false);
      } catch (err) {
        console.error('Failed to fetch schema data:', err);
        setError('Failed to load schema data: ' + (err.message || 'Unknown error'));
        setMappings([]);
        setConfirmedCount(0);
        setIsProcessing(false);
      }
    };
    
    // Simulate processing delay then fetch
    const timer = setTimeout(() => {
      fetchSchemaData();
    }, 1500);

    return () => clearTimeout(timer);
  }, [entityData, navigate]);

  const handleMappingChange = (id, newMapping) => {
    setMappings(prev => 
      prev.map(item => 
        item.id === id ? { ...item, mappedTo: newMapping, status: 'manual' } : item
      )
    );
    setEditingMapping(null);
  };

  const handleConfirmMapping = (id) => {
    setMappings(prev => 
      prev.map(item => 
        item.id === id ? { ...item, status: 'confirmed' } : item
      )
    );
  };

  const handleResetMapping = (id) => {
    const originalMapping = mockSchemaMapping.find(m => m.id === id);
    setMappings(prev => 
      prev.map(item => 
        item.id === id ? { ...item, mappedTo: originalMapping?.mappedTo, status: 'auto-mapped' } : item
      )
    );
  };

  const getStatusBadge = (status) => {
    switch (status) {
      case 'confirmed':
        return { bg: 'bg-accent/10', text: 'text-accent', label: 'Confirmed' };
      case 'manual':
        return { bg: 'bg-warning/10', text: 'text-warning', label: 'Modified' };
      case 'auto-mapped':
        return { bg: 'bg-primary/10', text: 'text-primary', label: 'Auto-Mapped' };
      default:
        return { bg: 'bg-gray-100', text: 'text-gray-600', label: 'Pending' };
    }
  };

  const getConfidenceColor = (confidence) => {
    if (confidence >= 90) return 'text-accent';
    if (confidence >= 75) return 'text-warning';
    return 'text-danger';
  };

  const getCategoryColor = (category) => {
    const colors = {
      'Financial Strength': 'bg-blue-100 text-blue-700',
      'Capital Structure': 'bg-purple-100 text-purple-700',
      'Repayment Capacity': 'bg-green-100 text-green-700',
      'Legal Risk': 'bg-red-100 text-red-700',
      'Reputation': 'bg-yellow-100 text-yellow-700',
      'Compliance': 'bg-indigo-100 text-indigo-700',
      'Market Outlook': 'bg-cyan-100 text-cyan-700',
      'Collateral': 'bg-orange-100 text-orange-700'
    };
    return colors[category] || 'bg-gray-100 text-gray-700';
  };

  const allConfirmed = mappings.every(m => m.status === 'confirmed');
  const progressPercent = mappings.length > 0 
    ? (mappings.filter(m => m.status === 'confirmed').length / mappings.length) * 100 
    : 0;

  const handleProceed = () => {
    navigate('/processing', { 
      state: { 
        entityData, 
        files, 
        classifications,
        mappings 
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
              onClick={() => navigate('/classification', { state: { entityData, files } })}
              className="p-2 hover:bg-gray-100 rounded-lg transition-colors"
            >
              <ArrowLeft className="w-5 h-5 text-gray-600" />
            </button>
            <div>
              <h1 className="text-2xl font-bold text-primary">Dynamic Schema Mapping</h1>
              <p className="text-gray-500 mt-1">Map extracted fields to structured credit analysis schema</p>
            </div>
          </div>

          {/* Processing State */}
          {isProcessing ? (
            <div className="bg-white rounded-xl border border-gray-100 shadow-sm p-12 text-center">
              <div className="inline-flex items-center justify-center w-20 h-20 bg-primary/10 rounded-2xl mb-6">
                <Database className="w-10 h-10 text-primary animate-pulse" />
              </div>
              <h2 className="text-xl font-semibold text-primary mb-2">Extracting & Mapping Data...</h2>
              <p className="text-gray-500 mb-6">AI is extracting financial entities and mapping to schema</p>
              <div className="max-w-md mx-auto">
                <div className="h-2 bg-gray-200 rounded-full overflow-hidden">
                  <div className="h-full bg-gradient-to-r from-primary to-secondary animate-pulse rounded-full" 
                       style={{ width: '70%' }} />
                </div>
              </div>
              <div className="mt-6 grid grid-cols-2 gap-4 max-w-md mx-auto text-left">
                <div className="flex items-center gap-2 text-sm text-gray-500">
                  <CheckCircle2 className="w-4 h-4 text-accent" />
                  <span>Parsing financial tables...</span>
                </div>
                <div className="flex items-center gap-2 text-sm text-gray-500">
                  <CheckCircle2 className="w-4 h-4 text-accent" />
                  <span>Identifying key metrics...</span>
                </div>
                <div className="flex items-center gap-2 text-sm text-gray-500">
                  <RefreshCw className="w-4 h-4 text-primary animate-spin" />
                  <span>Mapping to schema...</span>
                </div>
                <div className="flex items-center gap-2 text-sm text-gray-500">
                  <div className="w-4 h-4 rounded-full border-2 border-gray-300" />
                  <span>Validating mappings...</span>
                </div>
              </div>
            </div>
          ) : error ? (
            <div className="bg-warning/10 border border-warning/20 rounded-xl p-6 mb-6">
              <div className="flex items-center gap-3">
                <AlertCircle className="w-6 h-6 text-warning" />
                <div>
                  <h3 className="font-semibold text-warning">Warning</h3>
                  <p className="text-sm text-gray-600">{error}</p>
                </div>
              </div>
            </div>
          ) : (
            <>
              {/* Progress Summary */}
              <div className="bg-white rounded-xl border border-gray-100 shadow-sm p-6 mb-6">
                <div className="flex items-center justify-between">
                  <div className="flex items-center gap-4">
                    <div className="w-12 h-12 bg-primary/10 rounded-xl flex items-center justify-center">
                      <Settings2 className="w-6 h-6 text-primary" />
                    </div>
                    <div>
                      <h3 className="font-semibold text-primary">Schema Mapping Progress</h3>
                      <p className="text-sm text-gray-500">
                        {mappings.filter(m => m.status === 'confirmed').length} of {mappings.length} fields confirmed
                      </p>
                    </div>
                  </div>
                  <div className="text-right">
                    <p className="text-2xl font-bold text-primary">{Math.round(progressPercent)}%</p>
                    <p className="text-sm text-gray-500">Confirmed</p>
                  </div>
                </div>
                <div className="mt-4 h-2 bg-gray-100 rounded-full overflow-hidden">
                  <div 
                    className="h-full bg-accent rounded-full transition-all duration-500"
                    style={{ width: `${progressPercent}%` }}
                  />
                </div>
              </div>

              {/* Schema Mapping Table */}
              <div className="bg-white rounded-xl border border-gray-100 shadow-sm overflow-hidden">
                <div className="p-6 border-b border-gray-100">
                  <div className="flex items-center justify-between">
                    <div>
                      <h2 className="text-lg font-semibold text-primary">Field Mappings</h2>
                      <p className="text-sm text-gray-500 mt-1">
                        Review AI-extracted fields and their schema mappings. Edit if needed.
                      </p>
                    </div>
                    <button
                      onClick={() => setMappings(prev => prev.map(m => ({ ...m, status: 'confirmed' })))}
                      className="px-4 py-2 bg-accent text-white rounded-lg text-sm font-medium hover:bg-accent-dark transition-colors"
                    >
                      Confirm All
                    </button>
                  </div>
                </div>

                <div className="overflow-x-auto">
                  <table className="w-full">
                    <thead className="bg-gray-50">
                      <tr>
                        <th className="px-6 py-4 text-left text-xs font-semibold text-gray-500 uppercase tracking-wider">Extracted Field</th>
                        <th className="px-6 py-4 text-left text-xs font-semibold text-gray-500 uppercase tracking-wider">Detected Value</th>
                        <th className="px-6 py-4 text-left text-xs font-semibold text-gray-500 uppercase tracking-wider">Maps To Schema</th>
                        <th className="px-6 py-4 text-left text-xs font-semibold text-gray-500 uppercase tracking-wider">Category</th>
                        <th className="px-6 py-4 text-left text-xs font-semibold text-gray-500 uppercase tracking-wider">Confidence</th>
                        <th className="px-6 py-4 text-left text-xs font-semibold text-gray-500 uppercase tracking-wider">Status</th>
                        <th className="px-6 py-4 text-left text-xs font-semibold text-gray-500 uppercase tracking-wider">Actions</th>
                      </tr>
                    </thead>
                    <tbody className="divide-y divide-gray-100">
                      {mappings.map((item) => {
                        const statusBadge = getStatusBadge(item.status);
                        const mappedField = schemaFields.find(f => f.id === item.mappedTo);

                        return (
                          <tr key={item.id} className="hover:bg-gray-50 transition-colors">
                            <td className="px-6 py-4">
                              <div className="flex items-center gap-3">
                                <div className="w-8 h-8 bg-primary/10 rounded-lg flex items-center justify-center">
                                  <FileText className="w-4 h-4 text-primary" />
                                </div>
                                <span className="font-medium text-gray-900">{item.extractedField}</span>
                              </div>
                            </td>
                            <td className="px-6 py-4">
                              <span className="font-mono text-sm text-gray-700 bg-gray-100 px-2 py-1 rounded">
                                {item.detectedValue}
                              </span>
                            </td>
                            <td className="px-6 py-4">
                              {editingMapping === item.id ? (
                                <select
                                  value={item.mappedTo}
                                  onChange={(e) => handleMappingChange(item.id, e.target.value)}
                                  className="input-field text-sm py-1"
                                  autoFocus
                                >
                                  {schemaFields.map(field => (
                                    <option key={field.id} value={field.id}>{field.label}</option>
                                  ))}
                                </select>
                              ) : (
                                <div className="flex items-center gap-2">
                                  <Link2 className="w-4 h-4 text-accent" />
                                  <span className="font-medium text-gray-900">{mappedField?.label}</span>
                                </div>
                              )}
                            </td>
                            <td className="px-6 py-4">
                              <span className={`inline-flex items-center px-2.5 py-1 rounded-full text-xs font-medium ${getCategoryColor(mappedField?.category)}`}>
                                {mappedField?.category}
                              </span>
                            </td>
                            <td className="px-6 py-4">
                              <div className="flex items-center gap-2">
                                <div className="w-12 h-1.5 bg-gray-200 rounded-full overflow-hidden">
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
                              <span className={`inline-flex items-center px-2.5 py-1 rounded-full text-xs font-medium ${statusBadge.bg} ${statusBadge.text}`}>
                                {statusBadge.label}
                              </span>
                            </td>
                            <td className="px-6 py-4">
                              <div className="flex items-center gap-1">
                                {item.status !== 'confirmed' && (
                                  <button
                                    onClick={() => handleConfirmMapping(item.id)}
                                    className="p-2 text-accent hover:bg-accent/10 rounded-lg transition-colors"
                                    title="Confirm Mapping"
                                  >
                                    <CheckCircle2 className="w-4 h-4" />
                                  </button>
                                )}
                                <button
                                  onClick={() => setEditingMapping(item.id)}
                                  className="p-2 text-primary hover:bg-primary/10 rounded-lg transition-colors"
                                  title="Edit Mapping"
                                >
                                  <Settings2 className="w-4 h-4" />
                                </button>
                                {item.status === 'manual' && (
                                  <button
                                    onClick={() => handleResetMapping(item.id)}
                                    className="p-2 text-gray-500 hover:bg-gray-100 rounded-lg transition-colors"
                                    title="Reset to Auto"
                                  >
                                    <RefreshCw className="w-4 h-4" />
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

              {/* Schema Categories Legend */}
              <div className="mt-6 grid grid-cols-2 md:grid-cols-4 gap-3">
                {schemaFields.map((field) => (
                  <div 
                    key={field.id}
                    className={`flex items-center gap-2 px-3 py-2 rounded-lg text-xs ${getCategoryColor(field.category)}`}
                  >
                    <span className="font-medium">{field.label}</span>
                    <span className="opacity-70">• {field.category}</span>
                  </div>
                ))}
              </div>

              {/* Action Buttons */}
              <div className="flex items-center justify-between mt-8">
                <button
                  onClick={() => navigate('/classification', { state: { entityData, files } })}
                  className="flex items-center gap-2 px-6 py-3 border border-gray-200 text-gray-700 rounded-lg font-medium hover:bg-gray-50 transition-colors"
                >
                  <ArrowLeft className="w-4 h-4" />
                  Back to Classification
                </button>
                <button
                  onClick={handleProceed}
                  disabled={!allConfirmed}
                  className="flex items-center gap-2 px-8 py-3 bg-primary text-white rounded-lg font-medium hover:bg-primary-light transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
                >
                  Start AI Analysis
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

export default SchemaMapping;
