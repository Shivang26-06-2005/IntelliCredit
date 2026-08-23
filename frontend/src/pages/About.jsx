import { useState } from 'react';
import { 
  Brain, 
  Shield, 
  Zap, 
  FileSearch, 
  TrendingUp, 
  Users,
  CheckCircle2
} from 'lucide-react';
import Navbar from '../components/Navbar';
import Sidebar from '../components/Sidebar';

const features = [
  {
    icon: FileSearch,
    title: 'Document Intelligence',
    description: 'Extract and analyze data from PDFs, GST filings, bank statements, and legal notices using advanced OCR and NLP.'
  },
  {
    icon: Brain,
    title: 'AI-Powered Analysis',
    description: 'Machine learning models assess creditworthiness by analyzing financial patterns, cash flows, and risk indicators.'
  },
  {
    icon: Shield,
    title: 'Risk Detection',
    description: 'Identify early warning signals including circular trading, litigation risks, and GST mismatches automatically.'
  },
  {
    icon: Zap,
    title: 'Fast Processing',
    description: 'Reduce loan appraisal time from weeks to minutes with automated document processing and risk scoring.'
  },
  {
    icon: TrendingUp,
    title: 'Explainable AI',
    description: 'Transparent decision-making with clear explanations for risk scores and loan recommendations.'
  },
  {
    icon: Users,
    title: 'Research Agent',
    description: 'Automated web research for company news, regulatory updates, and litigation history from MCA and other sources.'
  }
];

const stats = [
  { value: '95%', label: 'Document Extraction Accuracy' },
  { value: '80%', label: 'Reduction in Processing Time' },
  { value: '3x', label: 'Better Risk Detection' },
  { value: '₹500Cr+', label: 'Loans Analyzed' }
];

const About = () => {
  const [sidebarOpen, setSidebarOpen] = useState(false);

  return (
    <div className="min-h-screen bg-background">
      <Navbar onMenuClick={() => setSidebarOpen(true)} />
      <Sidebar isOpen={sidebarOpen} onClose={() => setSidebarOpen(false)} />
      
      <main className="lg:ml-64 pt-16 min-h-screen">
        {/* Hero Section */}
        <div className="bg-gradient-to-br from-primary to-secondary py-16 px-6">
          <div className="max-w-4xl mx-auto text-center text-white">
            <div className="inline-flex items-center justify-center w-20 h-20 bg-white/20 rounded-2xl mb-6">
              <Brain className="w-10 h-10" />
            </div>
            <h1 className="text-4xl md:text-5xl font-bold mb-4">IntelliCredit AI</h1>
            <p className="text-xl text-white/80 mb-8">
              Revolutionizing Corporate Credit Appraisal with Artificial Intelligence
            </p>
            <div className="flex flex-wrap justify-center gap-4">
              <span className="px-4 py-2 bg-white/20 rounded-full text-sm">AI-Powered</span>
              <span className="px-4 py-2 bg-white/20 rounded-full text-sm">Explainable</span>
              <span className="px-4 py-2 bg-white/20 rounded-full text-sm">Indian Context</span>
            </div>
          </div>
        </div>

        <div className="p-6 max-w-6xl mx-auto -mt-8">
          {/* Stats */}
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-12">
            {stats.map((stat, index) => (
              <div key={index} className="bg-white rounded-xl p-6 text-center shadow-sm border border-gray-100">
                <p className="text-2xl md:text-3xl font-bold text-primary">{stat.value}</p>
                <p className="text-sm text-gray-500 mt-1">{stat.label}</p>
              </div>
            ))}
          </div>

          {/* Problem Statement */}
          <div className="mb-12">
            <h2 className="text-2xl font-bold text-primary mb-6">The Challenge in Corporate Lending</h2>
            <div className="bg-white rounded-xl p-8 border border-gray-100 shadow-sm">
              <div className="grid md:grid-cols-2 gap-8">
                <div>
                  <h3 className="text-lg font-semibold text-gray-900 mb-4">Current Pain Points</h3>
                  <ul className="space-y-3">
                    {[
                      'Weeks to process a single loan application',
                      'Manual analysis of hundreds of pages',
                      'Inability to detect hidden risk patterns',
                      'Inconsistent evaluation criteria',
                      'Limited access to external intelligence'
                    ].map((point, index) => (
                      <li key={index} className="flex items-start gap-3 text-gray-600">
                        <span className="w-1.5 h-1.5 bg-danger rounded-full mt-2 flex-shrink-0"></span>
                        {point}
                      </li>
                    ))}
                  </ul>
                </div>
                <div>
                  <h3 className="text-lg font-semibold text-gray-900 mb-4">Our Solution</h3>
                  <ul className="space-y-3">
                    {[
                      'AI processes documents in minutes',
                      'Automated extraction from any format',
                      'Advanced pattern recognition for risks',
                      'Standardized, explainable scoring',
                      'Integrated web research capabilities'
                    ].map((point, index) => (
                      <li key={index} className="flex items-start gap-3 text-gray-600">
                        <CheckCircle2 className="w-4 h-4 text-accent mt-0.5 flex-shrink-0" />
                        {point}
                      </li>
                    ))}
                  </ul>
                </div>
              </div>
            </div>
          </div>

          {/* Features */}
          <div className="mb-12">
            <h2 className="text-2xl font-bold text-primary mb-6 text-center">Key Features</h2>
            <div className="grid md:grid-cols-2 lg:grid-cols-3 gap-6">
              {features.map((feature, index) => {
                const Icon = feature.icon;
                return (
                  <div 
                    key={index} 
                    className="bg-white rounded-xl p-6 border border-gray-100 shadow-sm hover:shadow-md transition-shadow"
                  >
                    <div className="w-12 h-12 bg-primary/10 rounded-xl flex items-center justify-center mb-4">
                      <Icon className="w-6 h-6 text-primary" />
                    </div>
                    <h3 className="text-lg font-semibold text-primary mb-2">{feature.title}</h3>
                    <p className="text-gray-600 text-sm">{feature.description}</p>
                  </div>
                );
              })}
            </div>
          </div>

          {/* Architecture */}
          <div className="mb-12">
            <h2 className="text-2xl font-bold text-primary mb-6">System Architecture</h2>
            <div className="bg-white rounded-xl p-8 border border-gray-100 shadow-sm">
              <div className="grid md:grid-cols-5 gap-4 text-center">
                {[
                  { title: 'Data Sources', desc: 'PDFs, GST, Bank Statements, News' },
                  { title: 'Data Ingestor', desc: 'OCR + Parsing + Extraction' },
                  { title: 'AI Agents', desc: 'Financial, Research, Risk Detection' },
                  { title: 'Risk Model', desc: 'ML Scoring + Rule Engine' },
                  { title: 'CAM Generator', desc: 'Automated Report Generation' }
                ].map((item, index) => (
                  <div key={index} className="relative">
                    <div className="bg-primary/5 rounded-lg p-4 h-full">
                      <p className="font-semibold text-primary text-sm">{item.title}</p>
                      <p className="text-xs text-gray-500 mt-1">{item.desc}</p>
                    </div>
                    {index < 4 && (
                      <div className="hidden md:block absolute top-1/2 -right-2 w-4 h-px bg-gray-300"></div>
                    )}
                  </div>
                ))}
              </div>
            </div>
          </div>

          {/* CTA */}
          <div className="bg-gradient-to-r from-primary to-secondary rounded-2xl p-8 text-center text-white">
            <h2 className="text-2xl font-bold mb-4">Ready to Transform Your Credit Process?</h2>
            <p className="text-white/80 mb-6 max-w-2xl mx-auto">
              Join leading banks and financial institutions using IntelliCredit AI to make faster, 
              smarter, and more transparent credit decisions.
            </p>
            <button className="px-8 py-3 bg-white text-primary rounded-lg font-medium hover:bg-gray-100 transition-colors">
              Get Started Today
            </button>
          </div>
        </div>
      </main>
    </div>
  );
};

export default About;
