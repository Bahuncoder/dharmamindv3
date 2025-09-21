# 📊 Advanced Analytics Dashboard

## Overview

The Advanced Analytics Dashboard is a comprehensive real-time monitoring and analytics platform for the DharmaMind system. It provides deep insights into system performance, user behavior, and spiritual guidance effectiveness.

## Features

### 🔄 Real-time Data Streaming

- **WebSocket Integration**: Live data updates via WebSocket connections
- **Fallback Polling**: Automatic fallback to REST API polling if WebSocket fails
- **Connection Status**: Visual indicators for connection health
- **Auto-reconnection**: Automatic reconnection with exponential backoff

### 📈 System Metrics

- **Performance Monitoring**: CPU, memory, disk usage tracking
- **Response Time Analytics**: Real-time response time monitoring
- **Error Rate Tracking**: System error rate and alert generation
- **Active Connections**: Live connection count monitoring

### 👥 User Analytics

- **Active Users**: Real-time active user count
- **Session Analytics**: Session duration and engagement metrics
- **User Satisfaction**: Satisfaction score tracking
- **Growth Metrics**: New user acquisition and retention

### 🕉️ Dharma Insights

- **Wisdom Requests**: Spiritual guidance request tracking
- **Popular Topics**: Trending spiritual topics and themes
- **Guidance Effectiveness**: Effectiveness scoring for spiritual guidance
- **Satisfaction Metrics**: User satisfaction with dharmic guidance

### 📱 Mobile & Accessibility

- **Responsive Design**: Mobile-first responsive layouts
- **Touch Optimization**: Touch-friendly interactions
- **Screen Reader Support**: Full ARIA compliance
- **High Contrast Mode**: Accessibility-enhanced visuals
- **Reduced Motion**: Respect for motion preferences

## Components

### Core Components

#### `AdvancedAnalyticsDashboard.tsx`

Main dashboard component with comprehensive analytics visualization.

**Props:**

- `dashboardId`: Dashboard configuration identifier
- `className`: Additional CSS classes
- `refreshInterval`: Data refresh interval (default: 30s)
- `isAdmin`: Admin mode flag
- `isMobile`: Mobile optimization flag
- `reduceMotion`: Motion reduction flag
- `isHighContrast`: High contrast mode flag

#### `analytics.tsx`

Full-page dashboard with navigation and settings.

**Features:**

- Dashboard type switching
- Settings panel with preferences
- Mobile/desktop view toggles
- Accessibility controls

#### `useAdvancedAnalytics.ts`

Custom React hook for dashboard data management.

**Returns:**

- Real-time metrics data
- Connection status
- Data refresh functions
- Formatting utilities

### Styling

#### `analytics-dashboard.css`

Comprehensive CSS styles for the analytics dashboard.

**Features:**

- Mobile-responsive design
- Dark mode support
- High contrast accessibility
- Smooth animations and transitions
- Touch-friendly interactions

## Usage

### Basic Implementation

```tsx
import AdvancedAnalyticsDashboard from "../components/AdvancedAnalyticsDashboard";

function MyDashboard() {
  return (
    <AdvancedAnalyticsDashboard
      dashboardId="system_overview"
      refreshInterval={30000}
    />
  );
}
```

### With Custom Hook

```tsx
import { useAdvancedAnalytics } from "../hooks/useAdvancedAnalytics";

function CustomDashboard() {
  const { metrics, isConnected, connectionStatus, refresh, formatNumber } =
    useAdvancedAnalytics({
      dashboardId: "user_analytics",
      refreshInterval: 15000,
    });

  return (
    <div>
      <h1>Active Users: {formatNumber(metrics.user.active_users)}</h1>
      <p>Status: {connectionStatus}</p>
      <button onClick={refresh}>Refresh</button>
    </div>
  );
}
```

## Backend Integration

The dashboard connects to the DharmaMind backend analytics infrastructure:

- **WebSocket Endpoint**: `/api/dashboard/ws/{dashboard_id}`
- **REST API**: `/api/dashboard/{dashboard_id}`
- **Real-time Updates**: Automatic data streaming
- **Metrics Collection**: Integrated with existing monitoring systems

### Supported Dashboard Types

1. **`system_overview`**: System health and performance metrics
2. **`user_analytics`**: User behavior and engagement analysis
3. **`dharma_insights`**: Spiritual guidance effectiveness metrics

## Configuration

### Environment Variables

```env
NEXT_PUBLIC_WS_URL=ws://localhost:8000
NEXT_PUBLIC_API_URL=http://localhost:8000
```

### VS Code Setup

The project includes VS Code configuration for optimal development:

- **Tailwind CSS IntelliSense**: Proper CSS directive handling
- **TypeScript Support**: Enhanced TypeScript development
- **CSS Validation**: Configured to ignore Tailwind directives

### Tailwind CSS

The dashboard uses Tailwind CSS for styling:

```javascript
// tailwind.config.js
module.exports = {
  content: [
    "./pages/**/*.{js,ts,jsx,tsx,mdx}",
    "./components/**/*.{js,ts,jsx,tsx,mdx}",
  ],
  // ... configuration
};
```

## Accessibility

The dashboard is built with accessibility as a primary concern:

### WCAG 2.1 AA Compliance

- **Screen Reader Support**: Full ARIA labeling and announcements
- **Keyboard Navigation**: Complete keyboard accessibility
- **Color Contrast**: High contrast mode available
- **Motion Sensitivity**: Reduced motion respect

### Mobile Accessibility

- **Touch Targets**: Minimum 44px touch targets
- **Safe Areas**: Support for mobile safe areas
- **Haptic Feedback**: Touch feedback for supported devices
- **Virtual Keyboard**: Optimized keyboard handling

## Performance

### Optimization Features

- **Efficient Rendering**: React.memo and useMemo optimizations
- **WebSocket Management**: Proper connection lifecycle management
- **Data Caching**: Intelligent data caching and updates
- **Lazy Loading**: Component lazy loading where appropriate

### Monitoring

- **Performance Metrics**: Built-in performance tracking
- **Error Boundaries**: Comprehensive error handling
- **Memory Management**: Proper cleanup and garbage collection

## Development

### Getting Started

1. **Install Dependencies**:

   ```bash
   npm install
   ```

2. **Start Development Server**:

   ```bash
   npm run dev
   ```

3. **Build for Production**:
   ```bash
   npm run build
   ```

### Testing

The dashboard includes comprehensive testing:

```bash
# Run tests
npm test

# Run accessibility tests
npm run test:a11y

# Run performance tests
npm run test:perf
```

## Security

### Data Protection

- **Authentication**: Admin authentication for sensitive metrics
- **Authorization**: Role-based access control
- **Data Sanitization**: Input validation and sanitization
- **HTTPS Only**: Secure connection requirements

### Privacy

- **User Anonymization**: Personal data anonymization
- **Data Retention**: Configurable data retention policies
- **Audit Logging**: Comprehensive audit trail

## Troubleshooting

### Common Issues

1. **WebSocket Connection Failed**:

   - Check backend server status
   - Verify WebSocket endpoint availability
   - Check network connectivity

2. **Data Not Updating**:

   - Verify refresh interval settings
   - Check backend data source status
   - Review browser console for errors

3. **Performance Issues**:
   - Adjust refresh interval
   - Enable data caching
   - Check system resources

### Support

For technical support or feature requests, please refer to the main DharmaMind documentation or contact the development team.

---

## Integration with DharmaMind

This analytics dashboard is the **10th major improvement** in the DharmaMind Enhanced User Experience series, providing:

- **Real-time monitoring** of the spiritual guidance platform
- **Performance insights** for system optimization
- **User behavior analytics** for better spiritual guidance
- **Dharmic effectiveness** measurement and improvement

The dashboard seamlessly integrates with DharmaMind's consciousness-aware architecture while maintaining the platform's spiritual focus and dharmic principles.
