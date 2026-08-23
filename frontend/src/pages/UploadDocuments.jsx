import { useState } from 'react';
import { useNavigate, useLocation } from 'react-router-dom';
import { ArrowLeft, Upload, FileCheck, AlertCircle, Building2, Briefcase } from 'lucide-react';
import Navbar from '../components/Navbar';
import Sidebar from '../components/Sidebar';
import FileUpload from '../components/FileUpload';
import { uploadDocument } from '../services/api';

const UploadDocuments = () => {
  const navigate = useNavigate();
  const location = useLocation();
  const entityData = location.state?.entityData;
  
  const [sidebarOpen, setSidebarOpen] = useState(false);
  const [files, setFiles] = useState([]);
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [uploadStatus, setUploadStatus] = useState({});
  const [error, setError] = useState(null);
  const [uploadedDocs, setUploadedDocs] = useState([]);

  // Redirect if no entity data
  if (!entityData) {
    navigate('/onboarding');
    return null;
  }

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (files.length === 0) {
      alert('Please upload at least one document');
      return;
    }
    
    setIsSubmitting(true);
    setError(null);
    
    try {
      // Upload each file to backend
      const uploadedDocuments = [];
      
      for (let i = 0; i < files.length; i++) {
        const file = files[i];
        setUploadStatus(prev => ({ ...prev, [file.id]: 'uploading' }));
        
        try {
          // Upload to backend API
          const uploadedDoc = await uploadDocument(entityData.id, file.file);
          uploadedDocuments.push(uploadedDoc);
          
          setUploadStatus(prev => ({ ...prev, [file.id]: 'completed' }));
        } catch (err) {
          console.error(`Failed to upload ${file.name}:`, err);
          setUploadStatus(prev => ({ ...prev, [file.id]: 'error' }));
          setError(`Failed to upload ${file.name}: ${err.message}`);
        }
      }
      
      setUploadedDocs(uploadedDocuments);
      setIsSubmitting(false);
      
      // Navigate to classification after all uploads
      if (uploadedDocuments.length > 0) {
        navigate('/classification', { state: { entityData, files, uploadedDocs: uploadedDocuments } });
      }
    } catch (err) {
      console.error('Upload failed:', err);
      setError('Upload failed: ' + err.message);
      setIsSubmitting(false);
    }
  };

  const requiredDocuments = [
    { type: 'alm', label: 'ALM Statement', required: true },
    { type: 'shareholding', label: 'Shareholding Pattern', required: true },
    { type: 'borrowing', label: 'Borrowing Profile', required: true },
    { type: 'annual_report', label: 'Annual Report', required: true },
    { type: 'portfolio', label: 'Portfolio/Performance Data', required: false }
  ];

  const getUploadStatusIcon = (fileId) => {
    const status = uploadStatus[fileId];
    if (status === 'uploading') {
      return <div className="w-5 h-5 border-2 border-primary/30 border-t-primary rounded-full animate-spin" />;
    }
    if (status === 'completed') {
      return <FileCheck className="w-5 h-5 text-accent" />;
    }
    return null;
  };

  return (
    <div className="min-h-screen bg-background">
      <Navbar onMenuClick={() => setSidebarOpen(true)} />
      <Sidebar isOpen={sidebarOpen} onClose={() => setSidebarOpen(false)} />
      
      <main className="lg:ml-64 pt-16 min-h-screen">
        <div className="p-6 max-w-5xl mx-auto">
          {/* Header */}
          <div className="flex items-center gap-4 mb-8">
            <button 
              onClick={() => navigate('/onboarding')}
              className="p-2 hover:bg-gray-100 rounded-lg transition-colors"
            >
              <ArrowLeft className="w-5 h-5 text-gray-600" />
            </button>
            <div>
              <h1 className="text-2xl font-bold text-primary">Document Upload</h1>
              <p className="text-gray-500 mt-1">Upload required documents for {entityData?.companyName}</p>
            </div>
          </div>

          {/* Entity Summary Card */}
          <div className="bg-gradient-to-r from-primary to-secondary rounded-xl p-6 text-white mb-6">
            <div className="flex items-center gap-3 mb-4">
              <Building2 className="w-5 h-5" />
              <h2 className="font-semibold">Entity Summary</h2>
            </div>
            <div className="grid grid-cols-2 md:grid-cols-4 gap-4 text-sm">
              <div>
                <p className="text-white/60">Company</p>
                <p className="font-medium">{entityData?.companyName}</p>
              </div>
              <div>
                <p className="text-white/60">CIN</p>
                <p className="font-medium">{entityData?.cinNumber}</p>
              </div>
              <div>
                <p className="text-white/60">Loan Amount</p>
                <p className="font-medium">₹{parseInt(entityData?.loanAmount || 0).toLocaleString('en-IN')}</p>
              </div>
              <div>
                <p className="text-white/60">Loan Type</p>
                <p className="font-medium capitalize">{entityData?.loanType?.replace('_', ' ')}</p>
              </div>
            </div>
          </div>

          <form onSubmit={handleSubmit} className="space-y-6">
            {/* Required Documents Checklist */}
            <div className="bg-white rounded-xl border border-gray-100 shadow-sm p-6">
              <h2 className="text-lg font-semibold text-primary mb-4 flex items-center gap-2">
                <FileCheck className="w-5 h-5" />
                Required Documents
              </h2>
              <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
                {requiredDocuments.map((doc) => {
                  const hasFile = files.some(f => f.type === doc.type);
                  return (
                    <div 
                      key={doc.type}
                      className={`flex items-center gap-3 p-3 rounded-lg border ${
                        hasFile 
                          ? 'bg-accent/5 border-accent/20' 
                          : doc.required 
                            ? 'bg-warning/5 border-warning/20' 
                            : 'bg-gray-50 border-gray-200'
                      }`}
                    >
                      <div className={`w-8 h-8 rounded-lg flex items-center justify-center ${
                        hasFile ? 'bg-accent/10' : doc.required ? 'bg-warning/10' : 'bg-gray-200'
                      }`}>
                        {hasFile ? (
                          <FileCheck className="w-4 h-4 text-accent" />
                        ) : doc.required ? (
                          <AlertCircle className="w-4 h-4 text-warning" />
                        ) : (
                          <Upload className="w-4 h-4 text-gray-400" />
                        )}
                      </div>
                      <div className="flex-1">
                        <p className="text-sm font-medium text-gray-900">{doc.label}</p>
                        <p className="text-xs text-gray-500">
                          {hasFile ? 'Uploaded' : doc.required ? 'Required' : 'Optional'}
                        </p>
                      </div>
                      {doc.required && !hasFile && (
                        <span className="text-xs text-warning font-medium">Required</span>
                      )}
                    </div>
                  );
                })}
              </div>
            </div>

            {/* Document Upload */}
            <div className="bg-white rounded-xl border border-gray-100 shadow-sm p-6">
              <h2 className="text-lg font-semibold text-primary mb-2">Upload Documents</h2>
              <p className="text-sm text-gray-500 mb-6">
                Upload financial documents for AI analysis. We support PDF, Excel (.xlsx, .xls), and CSV files.
              </p>
              
              {/* Error Display */}
              {error && (
                <div className="mb-6 p-4 bg-danger/10 border border-danger/20 rounded-xl">
                  <div className="flex items-center gap-3">
                    <AlertCircle className="w-5 h-5 text-danger" />
                    <p className="text-sm text-gray-700">{error}</p>
                  </div>
                </div>
              )}
              
              <FileUpload files={files} onFilesChange={setFiles} />

              {/* Upload Status */}
              {files.length > 0 && Object.keys(uploadStatus).length > 0 && (
                <div className="mt-6 pt-6 border-t border-gray-100">
                  <h3 className="text-sm font-medium text-gray-700 mb-3">Upload Status</h3>
                  <div className="space-y-2">
                    {files.map((file) => (
                      <div key={file.id} className="flex items-center justify-between p-2 bg-gray-50 rounded-lg">
                        <span className="text-sm text-gray-600 truncate">{file.file.name}</span>
                        {getUploadStatusIcon(file.id)}
                      </div>
                    ))}
                  </div>
                </div>
              )}
            </div>

            {/* Submit Button */}
            <div className="flex items-center justify-end gap-4">
              <button
                type="button"
                onClick={() => navigate('/onboarding')}
                className="px-6 py-3 border border-gray-200 text-gray-700 rounded-lg font-medium hover:bg-gray-50 transition-colors"
              >
                Back
              </button>
              <button
                type="submit"
                disabled={isSubmitting || files.length === 0}
                className="px-8 py-3 bg-primary text-white rounded-lg font-medium hover:bg-primary-light transition-colors disabled:opacity-50 disabled:cursor-not-allowed flex items-center gap-2"
              >
                {isSubmitting ? (
                  <>
                    <div className="w-5 h-5 border-2 border-white/30 border-t-white rounded-full animate-spin"></div>
                    Uploading...
                  </>
                ) : (
                  <>
                    Continue to Classification
                    <ArrowLeft className="w-4 h-4 rotate-180" />
                  </>
                )}
              </button>
            </div>
          </form>
        </div>
      </main>
    </div>
  );
};

export default UploadDocuments;
