import { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { 
  Building2, 
  Briefcase, 
  IndianRupee, 
  Calendar, 
  FileText, 
  ChevronRight, 
  ChevronLeft, 
  CheckCircle2,
  Landmark,
  Clock,
  Percent,
  Hash,
  AlertCircle
} from 'lucide-react';
import Navbar from '../components/Navbar';
import Sidebar from '../components/Sidebar';
import { createEntity } from '../services/api';

const EntityOnboarding = () => {
  const navigate = useNavigate();
  const [sidebarOpen, setSidebarOpen] = useState(false);
  const [currentStep, setCurrentStep] = useState(1);
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [error, setError] = useState(null);
  
  const [formData, setFormData] = useState({
    // Step 1: Entity Details
    companyName: '',
    cinNumber: '',
    panNumber: '',
    industrySector: '',
    companyTurnover: '',
    // Step 2: Loan Details
    loanType: '',
    loanAmount: '',
    loanTenure: '',
    tenureUnit: 'years',
    expectedInterestRate: ''
  });

  const industries = [
    'Manufacturing',
    'IT Services',
    'Textiles',
    'Pharmaceuticals',
    'Automobile',
    'Real Estate',
    'Infrastructure',
    'Retail',
    'Banking & Finance',
    'Healthcare',
    'Education',
    'Others'
  ];

  const loanTypes = [
    { value: 'term_loan', label: 'Term Loan', description: 'Long-term financing for capital expenditure' },
    { value: 'working_capital', label: 'Working Capital', description: 'Short-term financing for operational needs' },
    { value: 'project_finance', label: 'Project Finance', description: 'Funding for specific projects' },
    { value: 'overdraft', label: 'Overdraft Facility', description: 'Flexible credit line for cash flow management' }
  ];

  const handleInputChange = (field, value) => {
    setFormData(prev => ({ ...prev, [field]: value }));
  };

  const validateStep = (step) => {
    if (step === 1) {
      return formData.companyName && formData.cinNumber && formData.panNumber && 
             formData.industrySector && formData.companyTurnover;
    }
    if (step === 2) {
      return formData.loanType && formData.loanAmount && formData.loanTenure && 
             formData.expectedInterestRate;
    }
    return true;
  };

  const handleNext = () => {
    if (validateStep(currentStep) && currentStep < 3) {
      setCurrentStep(currentStep + 1);
    }
  };

  const handleBack = () => {
    if (currentStep > 1) {
      setCurrentStep(currentStep - 1);
    }
  };

  const handleSubmit = async () => {
    setIsSubmitting(true);
    setError(null);
    
    try {
      // Create entity via backend API
      const entityData = {
        company_name: formData.companyName,
        cin: formData.cinNumber,
        pan: formData.panNumber,
        sector: formData.industrySector,
        annual_turnover: parseFloat(formData.companyTurnover) || 0,
        loan_amount: parseFloat(formData.loanAmount) || 0,
        loan_type: formData.loanType,
        loan_tenure: parseInt(formData.loanTenure) || 0
      };
      
      const createdEntity = await createEntity(entityData);
      
      setIsSubmitting(false);
      navigate('/upload', { 
        state: { 
          entityData: createdEntity,
          loanData: {
            loanType: formData.loanType,
            loanAmount: parseFloat(formData.loanAmount) || 0,
            tenureMonths: (parseInt(formData.loanTenure) || 0) * 12,
            interestRate: parseFloat(formData.expectedInterestRate) || 0
          }
        } 
      });
    } catch (err) {
      console.error('Failed to create entity:', err);
      setError(err.message || 'Failed to create entity. Please try again.');
      setIsSubmitting(false);
    }
  };

  const steps = [
    { number: 1, title: 'Entity Details', description: 'Company information' },
    { number: 2, title: 'Loan Details', description: 'Financing requirements' },
    { number: 3, title: 'Review', description: 'Verify information' }
  ];

  const formatCurrency = (value) => {
    if (!value) return '';
    const num = parseFloat(value.replace(/[^0-9]/g, ''));
    if (isNaN(num)) return value;
    return '₹' + num.toLocaleString('en-IN');
  };

  return (
    <div className="min-h-screen bg-background">
      <Navbar onMenuClick={() => setSidebarOpen(true)} />
      <Sidebar isOpen={sidebarOpen} onClose={() => setSidebarOpen(false)} />
      
      <main className="lg:ml-64 pt-16 min-h-screen">
        <div className="p-6 max-w-4xl mx-auto">
          {/* Header */}
          <div className="mb-8">
            <h1 className="text-2xl font-bold text-primary">Entity Onboarding</h1>
            <p className="text-gray-500 mt-1">Register a new borrower for credit analysis</p>
          </div>

          {/* Progress Steps */}
          <div className="mb-8">
            <div className="flex items-center justify-between">
              {steps.map((step, index) => (
                <div key={step.number} className="flex items-center flex-1">
                  <div className="flex flex-col items-center">
                    <div className={`w-10 h-10 rounded-full flex items-center justify-center font-semibold transition-all ${
                      currentStep >= step.number 
                        ? 'bg-primary text-white' 
                        : 'bg-gray-200 text-gray-500'
                    }`}>
                      {currentStep > step.number ? (
                        <CheckCircle2 className="w-5 h-5" />
                      ) : (
                        step.number
                      )}
                    </div>
                    <div className="text-center mt-2">
                      <p className={`text-sm font-medium ${
                        currentStep >= step.number ? 'text-primary' : 'text-gray-500'
                      }`}>
                        {step.title}
                      </p>
                      <p className="text-xs text-gray-400">{step.description}</p>
                    </div>
                  </div>
                  {index < steps.length - 1 && (
                    <div className={`flex-1 h-1 mx-4 transition-all ${
                      currentStep > step.number ? 'bg-primary' : 'bg-gray-200'
                    }`} />
                  )}
                </div>
              ))}
            </div>
          </div>

          {/* Form Content */}
          <div className="bg-white rounded-xl border border-gray-100 shadow-sm p-8">
            {/* Step 1: Entity Details */}
            {currentStep === 1 && (
              <div className="space-y-6">
                <div className="flex items-center gap-3 mb-6">
                  <div className="w-10 h-10 bg-primary/10 rounded-lg flex items-center justify-center">
                    <Building2 className="w-5 h-5 text-primary" />
                  </div>
                  <div>
                    <h2 className="text-lg font-semibold text-primary">Entity Details</h2>
                    <p className="text-sm text-gray-500">Enter the company registration information</p>
                  </div>
                </div>

                <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                  <div className="md:col-span-2">
                    <label className="block text-sm font-medium text-gray-700 mb-2">
                      Company Name *
                    </label>
                    <input
                      type="text"
                      value={formData.companyName}
                      onChange={(e) => handleInputChange('companyName', e.target.value)}
                      placeholder="e.g., ABC Manufacturing Ltd"
                      className="input-field"
                      required
                    />
                  </div>

                  <div>
                    <label className="block text-sm font-medium text-gray-700 mb-2">
                      CIN Number *
                    </label>
                    <div className="relative">
                      <Hash className="absolute left-3 top-1/2 -translate-y-1/2 w-5 h-5 text-gray-400" />
                      <input
                        type="text"
                        value={formData.cinNumber}
                        onChange={(e) => handleInputChange('cinNumber', e.target.value.toUpperCase())}
                        placeholder="L12345AB6789CDE123456"
                        className="input-field pl-10 uppercase"
                        maxLength={21}
                        required
                      />
                    </div>
                    <p className="text-xs text-gray-400 mt-1">Corporate Identification Number</p>
                  </div>

                  <div>
                    <label className="block text-sm font-medium text-gray-700 mb-2">
                      PAN Number *
                    </label>
                    <div className="relative">
                      <FileText className="absolute left-3 top-1/2 -translate-y-1/2 w-5 h-5 text-gray-400" />
                      <input
                        type="text"
                        value={formData.panNumber}
                        onChange={(e) => handleInputChange('panNumber', e.target.value.toUpperCase())}
                        placeholder="ABCDE1234F"
                        className="input-field pl-10 uppercase"
                        maxLength={10}
                        required
                      />
                    </div>
                    <p className="text-xs text-gray-400 mt-1">Permanent Account Number</p>
                  </div>

                  <div>
                    <label className="block text-sm font-medium text-gray-700 mb-2">
                      Industry Sector *
                    </label>
                    <div className="relative">
                      <Briefcase className="absolute left-3 top-1/2 -translate-y-1/2 w-5 h-5 text-gray-400" />
                      <select
                        value={formData.industrySector}
                        onChange={(e) => handleInputChange('industrySector', e.target.value)}
                        className="input-field pl-10 appearance-none bg-white"
                        required
                      >
                        <option value="">Select Industry</option>
                        {industries.map(industry => (
                          <option key={industry} value={industry}>{industry}</option>
                        ))}
                      </select>
                    </div>
                  </div>

                  <div>
                    <label className="block text-sm font-medium text-gray-700 mb-2">
                      Company Turnover (Annual) *
                    </label>
                    <div className="relative">
                      <IndianRupee className="absolute left-3 top-1/2 -translate-y-1/2 w-5 h-5 text-gray-400" />
                      <input
                        type="text"
                        value={formData.companyTurnover}
                        onChange={(e) => handleInputChange('companyTurnover', e.target.value)}
                        placeholder="e.g., 50,00,00,000"
                        className="input-field pl-10"
                        required
                      />
                    </div>
                  </div>
                </div>
              </div>
            )}

            {/* Step 2: Loan Details */}
            {currentStep === 2 && (
              <div className="space-y-6">
                <div className="flex items-center gap-3 mb-6">
                  <div className="w-10 h-10 bg-primary/10 rounded-lg flex items-center justify-center">
                    <Landmark className="w-5 h-5 text-primary" />
                  </div>
                  <div>
                    <h2 className="text-lg font-semibold text-primary">Loan Details</h2>
                    <p className="text-sm text-gray-500">Specify the financing requirements</p>
                  </div>
                </div>

                <div className="space-y-6">
                  <div>
                    <label className="block text-sm font-medium text-gray-700 mb-3">
                      Loan Type *
                    </label>
                    <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                      {loanTypes.map((type) => (
                        <button
                          key={type.value}
                          type="button"
                          onClick={() => handleInputChange('loanType', type.value)}
                          className={`p-4 rounded-xl border-2 text-left transition-all ${
                            formData.loanType === type.value
                              ? 'border-primary bg-primary/5'
                              : 'border-gray-200 hover:border-primary/50'
                          }`}
                        >
                          <p className="font-semibold text-primary">{type.label}</p>
                          <p className="text-sm text-gray-500 mt-1">{type.description}</p>
                        </button>
                      ))}
                    </div>
                  </div>

                  <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                    <div>
                      <label className="block text-sm font-medium text-gray-700 mb-2">
                        Loan Amount Requested *
                      </label>
                      <div className="relative">
                        <IndianRupee className="absolute left-3 top-1/2 -translate-y-1/2 w-5 h-5 text-gray-400" />
                        <input
                          type="text"
                          value={formData.loanAmount}
                          onChange={(e) => handleInputChange('loanAmount', e.target.value)}
                          placeholder="e.g., 10,00,00,000"
                          className="input-field pl-10"
                          required
                        />
                      </div>
                    </div>

                    <div>
                      <label className="block text-sm font-medium text-gray-700 mb-2">
                        Expected Interest Rate (%) *
                      </label>
                      <div className="relative">
                        <Percent className="absolute left-3 top-1/2 -translate-y-1/2 w-5 h-5 text-gray-400" />
                        <input
                          type="number"
                          step="0.1"
                          value={formData.expectedInterestRate}
                          onChange={(e) => handleInputChange('expectedInterestRate', e.target.value)}
                          placeholder="e.g., 12.5"
                          className="input-field pl-10"
                          required
                        />
                      </div>
                    </div>

                    <div>
                      <label className="block text-sm font-medium text-gray-700 mb-2">
                        Loan Tenure *
                      </label>
                      <div className="flex gap-3">
                        <div className="relative flex-1">
                          <Clock className="absolute left-3 top-1/2 -translate-y-1/2 w-5 h-5 text-gray-400" />
                          <input
                            type="number"
                            value={formData.loanTenure}
                            onChange={(e) => handleInputChange('loanTenure', e.target.value)}
                            placeholder="e.g., 5"
                            className="input-field pl-10"
                            required
                          />
                        </div>
                        <select
                          value={formData.tenureUnit}
                          onChange={(e) => handleInputChange('tenureUnit', e.target.value)}
                          className="input-field w-32 appearance-none bg-white"
                        >
                          <option value="years">Years</option>
                          <option value="months">Months</option>
                        </select>
                      </div>
                    </div>
                  </div>
                </div>
              </div>
            )}

            {/* Step 3: Review */}
            {currentStep === 3 && (
              <div className="space-y-6">
                <div className="flex items-center gap-3 mb-6">
                  <div className="w-10 h-10 bg-accent/10 rounded-lg flex items-center justify-center">
                    <CheckCircle2 className="w-5 h-5 text-accent" />
                  </div>
                  <div>
                    <h2 className="text-lg font-semibold text-primary">Review Information</h2>
                    <p className="text-sm text-gray-500">Verify all details before proceeding</p>
                  </div>
                </div>

                <div className="space-y-6">
                  {/* Entity Details Summary */}
                  <div className="bg-gray-50 rounded-xl p-6">
                    <div className="flex items-center justify-between mb-4">
                      <h3 className="font-semibold text-primary flex items-center gap-2">
                        <Building2 className="w-4 h-4" />
                        Entity Details
                      </h3>
                      <button 
                        onClick={() => setCurrentStep(1)}
                        className="text-sm text-primary hover:underline"
                      >
                        Edit
                      </button>
                    </div>
                    <div className="grid grid-cols-2 gap-4 text-sm">
                      <div>
                        <p className="text-gray-500">Company Name</p>
                        <p className="font-medium text-gray-900">{formData.companyName}</p>
                      </div>
                      <div>
                        <p className="text-gray-500">CIN Number</p>
                        <p className="font-medium text-gray-900">{formData.cinNumber}</p>
                      </div>
                      <div>
                        <p className="text-gray-500">PAN Number</p>
                        <p className="font-medium text-gray-900">{formData.panNumber}</p>
                      </div>
                      <div>
                        <p className="text-gray-500">Industry Sector</p>
                        <p className="font-medium text-gray-900">{formData.industrySector}</p>
                      </div>
                      <div className="col-span-2">
                        <p className="text-gray-500">Annual Turnover</p>
                        <p className="font-medium text-gray-900">{formatCurrency(formData.companyTurnover)}</p>
                      </div>
                    </div>
                  </div>

                  {/* Loan Details Summary */}
                  <div className="bg-gray-50 rounded-xl p-6">
                    <div className="flex items-center justify-between mb-4">
                      <h3 className="font-semibold text-primary flex items-center gap-2">
                        <Landmark className="w-4 h-4" />
                        Loan Details
                      </h3>
                      <button 
                        onClick={() => setCurrentStep(2)}
                        className="text-sm text-primary hover:underline"
                      >
                        Edit
                      </button>
                    </div>
                    <div className="grid grid-cols-2 gap-4 text-sm">
                      <div>
                        <p className="text-gray-500">Loan Type</p>
                        <p className="font-medium text-gray-900">
                          {loanTypes.find(t => t.value === formData.loanType)?.label}
                        </p>
                      </div>
                      <div>
                        <p className="text-gray-500">Loan Amount</p>
                        <p className="font-medium text-gray-900">{formatCurrency(formData.loanAmount)}</p>
                      </div>
                      <div>
                        <p className="text-gray-500">Loan Tenure</p>
                        <p className="font-medium text-gray-900">
                          {formData.loanTenure} {formData.tenureUnit}
                        </p>
                      </div>
                      <div>
                        <p className="text-gray-500">Expected Interest Rate</p>
                        <p className="font-medium text-gray-900">{formData.expectedInterestRate}%</p>
                      </div>
                    </div>
                  </div>
                </div>
              </div>
            )}

            {/* Navigation Buttons */}
            {/* Error Display */}
            {error && (
              <div className="mb-6 p-4 bg-danger/10 border border-danger/20 rounded-xl">
                <div className="flex items-center gap-3">
                  <AlertCircle className="w-5 h-5 text-danger" />
                  <p className="text-sm text-gray-700">{error}</p>
                </div>
              </div>
            )}
            
            <div className="flex items-center justify-between mt-8 pt-6 border-t border-gray-100">
              <button
                type="button"
                onClick={handleBack}
                disabled={currentStep === 1}
                className="flex items-center gap-2 px-6 py-3 border border-gray-200 text-gray-700 rounded-lg font-medium hover:bg-gray-50 transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
              >
                <ChevronLeft className="w-4 h-4" />
                Back
              </button>

              {currentStep < 3 ? (
                <button
                  type="button"
                  onClick={handleNext}
                  disabled={!validateStep(currentStep)}
                  className="flex items-center gap-2 px-8 py-3 bg-primary text-white rounded-lg font-medium hover:bg-primary-light transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
                >
                  Continue
                  <ChevronRight className="w-4 h-4" />
                </button>
              ) : (
                <button
                  type="button"
                  onClick={handleSubmit}
                  disabled={isSubmitting}
                  className="flex items-center gap-2 px-8 py-3 bg-accent text-white rounded-lg font-medium hover:bg-accent-dark transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
                >
                  {isSubmitting ? (
                    <>
                      <div className="w-5 h-5 border-2 border-white/30 border-t-white rounded-full animate-spin"></div>
                      Processing...
                    </>
                  ) : (
                    <>
                      Proceed to Document Upload
                      <ChevronRight className="w-4 h-4" />
                    </>
                  )}
                </button>
              )}
            </div>
          </div>
        </div>
      </main>
    </div>
  );
};

export default EntityOnboarding;
