import { useState } from 'react';
import { Mail, Phone, MapPin, Send, CheckCircle } from 'lucide-react';
import Navbar from '../components/Navbar';
import Sidebar from '../components/Sidebar';

const Contact = () => {
  const [sidebarOpen, setSidebarOpen] = useState(false);
  const [formData, setFormData] = useState({
    name: '',
    email: '',
    subject: '',
    message: ''
  });
  const [isSubmitted, setIsSubmitted] = useState(false);
  const [isSubmitting, setIsSubmitting] = useState(false);

  const handleSubmit = async (e) => {
    e.preventDefault();
    setIsSubmitting(true);
    
    // Simulate form submission
    setTimeout(() => {
      setIsSubmitting(false);
      setIsSubmitted(true);
      setFormData({ name: '', email: '', subject: '', message: '' });
    }, 1500);
  };

  return (
    <div className="min-h-screen bg-background">
      <Navbar onMenuClick={() => setSidebarOpen(true)} />
      <Sidebar isOpen={sidebarOpen} onClose={() => setSidebarOpen(false)} />
      
      <main className="lg:ml-64 pt-16 min-h-screen">
        <div className="p-6 max-w-6xl mx-auto">
          {/* Header */}
          <div className="text-center mb-12">
            <h1 className="text-3xl font-bold text-primary mb-4">Contact Us</h1>
            <p className="text-gray-500 max-w-2xl mx-auto">
              Have questions about IntelliCredit AI? We are here to help. Reach out to our team for support, 
              demos, or partnership inquiries.
            </p>
          </div>

          <div className="grid lg:grid-cols-3 gap-8">
            {/* Contact Info */}
            <div className="lg:col-span-1 space-y-6">
              <div className="bg-white rounded-xl p-6 border border-gray-100 shadow-sm">
                <div className="w-12 h-12 bg-primary/10 rounded-xl flex items-center justify-center mb-4">
                  <Mail className="w-6 h-6 text-primary" />
                </div>
                <h3 className="font-semibold text-primary mb-2">Email</h3>
                <p className="text-gray-600 text-sm">support@intellicredit.ai</p>
                <p className="text-gray-600 text-sm">sales@intellicredit.ai</p>
              </div>

              <div className="bg-white rounded-xl p-6 border border-gray-100 shadow-sm">
                <div className="w-12 h-12 bg-primary/10 rounded-xl flex items-center justify-center mb-4">
                  <Phone className="w-6 h-6 text-primary" />
                </div>
                <h3 className="font-semibold text-primary mb-2">Phone</h3>
                <p className="text-gray-600 text-sm">+91 1800-123-4567</p>
                <p className="text-gray-500 text-xs mt-1">Mon-Fri, 9AM-6PM IST</p>
              </div>

              <div className="bg-white rounded-xl p-6 border border-gray-100 shadow-sm">
                <div className="w-12 h-12 bg-primary/10 rounded-xl flex items-center justify-center mb-4">
                  <MapPin className="w-6 h-6 text-primary" />
                </div>
                <h3 className="font-semibold text-primary mb-2">Address</h3>
                <p className="text-gray-600 text-sm">
                  IntelliCredit Technologies Pvt. Ltd.<br />
                  Tech Park, Block C<br />
                  Hyderabad, Telangana 500081<br />
                  India
                </p>
              </div>
            </div>

            {/* Contact Form */}
            <div className="lg:col-span-2">
              <div className="bg-white rounded-xl border border-gray-100 shadow-sm p-8">
                {isSubmitted ? (
                  <div className="text-center py-12">
                    <div className="w-16 h-16 bg-accent/10 rounded-full flex items-center justify-center mx-auto mb-4">
                      <CheckCircle className="w-8 h-8 text-accent" />
                    </div>
                    <h3 className="text-xl font-semibold text-primary mb-2">Message Sent!</h3>
                    <p className="text-gray-600 mb-6">
                      Thank you for reaching out. We will get back to you within 24 hours.
                    </p>
                    <button
                      onClick={() => setIsSubmitted(false)}
                      className="px-6 py-2 bg-primary text-white rounded-lg hover:bg-primary-light transition-colors"
                    >
                      Send Another Message
                    </button>
                  </div>
                ) : (
                  <form onSubmit={handleSubmit} className="space-y-6">
                    <div className="grid md:grid-cols-2 gap-6">
                      <div>
                        <label className="block text-sm font-medium text-gray-700 mb-2">
                          Your Name *
                        </label>
                        <input
                          type="text"
                          value={formData.name}
                          onChange={(e) => setFormData({ ...formData, name: e.target.value })}
                          placeholder="John Doe"
                          className="input-field"
                          required
                        />
                      </div>
                      <div>
                        <label className="block text-sm font-medium text-gray-700 mb-2">
                          Email Address *
                        </label>
                        <input
                          type="email"
                          value={formData.email}
                          onChange={(e) => setFormData({ ...formData, email: e.target.value })}
                          placeholder="john@example.com"
                          className="input-field"
                          required
                        />
                      </div>
                    </div>

                    <div>
                      <label className="block text-sm font-medium text-gray-700 mb-2">
                        Subject *
                      </label>
                      <select
                        value={formData.subject}
                        onChange={(e) => setFormData({ ...formData, subject: e.target.value })}
                        className="input-field appearance-none bg-white"
                        required
                      >
                        <option value="">Select a subject</option>
                        <option value="general">General Inquiry</option>
                        <option value="demo">Request Demo</option>
                        <option value="support">Technical Support</option>
                        <option value="partnership">Partnership</option>
                        <option value="other">Other</option>
                      </select>
                    </div>

                    <div>
                      <label className="block text-sm font-medium text-gray-700 mb-2">
                        Message *
                      </label>
                      <textarea
                        value={formData.message}
                        onChange={(e) => setFormData({ ...formData, message: e.target.value })}
                        placeholder="How can we help you?"
                        rows={5}
                        className="input-field resize-none"
                        required
                      />
                    </div>

                    <button
                      type="submit"
                      disabled={isSubmitting}
                      className="w-full md:w-auto px-8 py-3 bg-primary text-white rounded-lg font-medium hover:bg-primary-light transition-colors disabled:opacity-70 disabled:cursor-not-allowed flex items-center justify-center gap-2"
                    >
                      {isSubmitting ? (
                        <>
                          <div className="w-5 h-5 border-2 border-white/30 border-t-white rounded-full animate-spin"></div>
                          Sending...
                        </>
                      ) : (
                        <>
                          <Send className="w-4 h-4" />
                          Send Message
                        </>
                      )}
                    </button>
                  </form>
                )}
              </div>
            </div>
          </div>
        </div>
      </main>
    </div>
  );
};

export default Contact;
