import React from 'react';
import { useRouter } from 'next/router';
import Link from 'next/link';

interface MobileNavigationProps {
  className?: string;
}

const MobileNavigation: React.FC<MobileNavigationProps> = ({ className = '' }) => {
  const router = useRouter();
  
  const navItems = [
    {
      id: 'chat',
      label: 'Chat',
      icon: '💬',
      href: '/chat',
      isActive: router.pathname === '/mobile-chat' || router.pathname === '/chat'
    },
    {
      id: 'conversations',
      label: 'History',
      icon: '📚',
      href: '/conversations',
      isActive: router.pathname === '/conversations'
    },
    {
      id: 'wisdom',
      label: 'Wisdom',
      icon: '🕉️',
      href: '/wisdom',
      isActive: router.pathname === '/wisdom'
    },
    {
      id: 'profile',
      label: 'Profile',
      icon: '👤',
      href: '/profile',
      isActive: router.pathname === '/profile'
    },
    {
      id: 'settings',
      label: 'Settings',
      icon: '⚙️',
      href: '/settings',
      isActive: router.pathname === '/settings'
    }
  ];

  return (
    <nav className={`mobile-nav ${className}`}>
      <div className="flex justify-around items-center max-w-md mx-auto">
        {navItems.map((item) => (
          <Link key={item.id} href={item.href}>
            <a className={`mobile-nav-item ${item.isActive ? 'active' : ''}`}>
              <span className="text-xl mb-1" role="img" aria-hidden="true">
                {item.icon}
              </span>
              <span className="text-xs">{item.label}</span>
            </a>
          </Link>
        ))}
      </div>
    </nav>
  );
};

export default MobileNavigation;
