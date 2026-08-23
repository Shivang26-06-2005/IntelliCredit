import { NavLink, useLocation, useNavigate } from 'react-router-dom';
import { 
  LayoutDashboard, 
  FilePlus, 
  FileText, 
  BarChart3, 
  Info, 
  Phone, 
  LogOut,
  X
} from 'lucide-react';
import { cn } from '../lib/utils';

const menuItems = [
  { path: '/', icon: LayoutDashboard, label: 'Dashboard' },
  { path: '/onboarding', icon: FilePlus, label: 'New Loan Analysis' },
  { path: '/reports', icon: BarChart3, label: 'Reports' },
  { path: '/cam-reports', icon: FileText, label: 'CAM Reports' },
  { path: '/about', icon: Info, label: 'About' },
  { path: '/contact', icon: Phone, label: 'Contact' },
];

const Sidebar = ({ isOpen, onClose }) => {
  const location = useLocation();
  const navigate = useNavigate();

  const handleLogout = () => {
    // Clear authentication data from localStorage
    localStorage.removeItem('isAuthenticated');
    localStorage.removeItem('userEmail');
    localStorage.removeItem('loginTime');
    
    // Redirect to login page
    navigate('/login', { replace: true });
  };

  return (
    <>
      {/* Mobile Overlay */}
      {isOpen && (
        <div 
          className="fixed inset-0 bg-black/50 z-40 lg:hidden"
          onClick={onClose}
        />
      )}

      {/* Sidebar */}
      <aside className={cn(
        "fixed left-0 top-16 h-[calc(100vh-4rem)] w-64 bg-primary border-r border-primary-light z-40 transition-transform duration-300 lg:translate-x-0",
        isOpen ? "translate-x-0" : "-translate-x-full"
      )}>
        {/* Close button for mobile */}
        <button 
          onClick={onClose}
          className="absolute top-4 right-4 lg:hidden text-white/70 hover:text-white"
        >
          <X className="w-5 h-5" />
        </button>

        <nav className="p-4 pt-6 lg:pt-4">
          <ul className="space-y-1">
            {menuItems.map((item) => {
              const Icon = item.icon;
              const isActive = location.pathname === item.path;
              
              return (
                <li key={item.path}>
                  <NavLink
                    to={item.path}
                    onClick={onClose}
                    className={cn(
                      "flex items-center gap-3 px-4 py-3 rounded-lg transition-all duration-200",
                      isActive 
                        ? "bg-white/10 text-white" 
                        : "text-white/70 hover:bg-white/5 hover:text-white"
                    )}
                  >
                    <Icon className={cn(
                      "w-5 h-5",
                      isActive ? "text-accent" : "text-white/70"
                    )} />
                    <span className="font-medium">{item.label}</span>
                    {isActive && (
                      <div className="ml-auto w-1.5 h-1.5 rounded-full bg-accent" />
                    )}
                  </NavLink>
                </li>
              );
            })}
          </ul>

          {/* Logout at bottom */}
          <div className="absolute bottom-6 left-4 right-4">
            <button 
              onClick={handleLogout}
              className="flex items-center gap-3 px-4 py-3 w-full rounded-lg text-white/70 hover:bg-white/5 hover:text-white transition-all duration-200"
            >
              <LogOut className="w-5 h-5" />
              <span className="font-medium">Logout</span>
            </button>
          </div>
        </nav>
      </aside>
    </>
  );
};

export default Sidebar;
