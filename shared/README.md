# 🧩 Shared Components Library

Reusable components, utilities, and design system for DharmaMind platform.

## 📁 Structure

```
shared/
├── components/          # Reusable React components
│   ├── dharma-components.jsx  # Core spiritual UI components
│   └── ...             # Additional shared components
├── styles/             # Shared CSS and themes
│   ├── tailwind.dharma.config.js  # Dharma design system config
│   └── ...             # Theme files, variables
├── utils/              # Shared utility functions
│   ├── dharma-design-system.js    # Design system utilities
│   └── ...             # Helper functions
├── types/              # TypeScript type definitions
└── hooks/              # Shared React hooks
```

## 🎨 Design System

### Dharma Color Palette

Our design system is built around spiritual and calming colors:

```javascript
// From tailwind.dharma.config.js
const dharmaColors = {
  saffron: '#FF9933',     // Sacred saffron
  lotus: '#FFB6C1',       // Lotus pink
  emerald: '#50C878',     # Peaceful green
  sapphire: '#0F52BA',    // Wisdom blue
  sandalwood: '#F4A460',  // Warm wood tone
}
```

### Typography

- **Headers**: Spiritual, calming fonts
- **Body**: Clean, readable fonts for long-form content
- **Accents**: Traditional Devanagari support

## 🧩 Component Usage

### Dharma Components

```jsx
import {
  DharmaButton,
  SpiritualCard,
  MeditationTimer,
} from "../shared/components/dharma-components";

// Usage in any frontend app
<DharmaButton variant="lotus" onClick={handleSpiritual}>
  Begin Meditation
</DharmaButton>;
```

### Design System Utilities

```javascript
import {
  getDharmaColor,
  getSpacingUnit,
} from "../shared/utils/dharma-design-system";

const primaryColor = getDharmaColor("saffron");
const spacing = getSpacingUnit("meditation"); // Returns appropriate spacing
```

## 🔧 Integration

### In Brand Website

```javascript
// Brand_Webpage/tailwind.config.js
const sharedConfig = require("../shared/styles/tailwind.dharma.config.js");

module.exports = {
  ...sharedConfig,
  // Brand-specific overrides
};
```

### In Chat Application

```javascript
// dharmamind-chat/tailwind.config.js
const sharedConfig = require("../shared/styles/tailwind.dharma.config.js");

module.exports = {
  ...sharedConfig,
  // Chat-specific overrides
};
```

### In Community Platform

```javascript
// DhramaMind_Community/tailwind.config.js
const sharedConfig = require("../shared/styles/tailwind.dharma.config.js");

module.exports = {
  ...sharedConfig,
  // Community-specific overrides
};
```

## 🎯 Benefits

### Consistency

- Unified design language across all applications
- Consistent component behavior and styling
- Shared interaction patterns

### Efficiency

- Reduce duplicate code across frontends
- Faster development of new features
- Easier maintenance and updates

### Scalability

- Easy to add new applications using existing components
- Centralized design system management
- Version-controlled component library

## 📦 Component Categories

### Core Components

- `DharmaButton` - Spiritually-themed buttons
- `SpiritualCard` - Content cards with dharmic styling
- `MeditationTimer` - Timer component for spiritual practices

### Layout Components

- `DharmaLayout` - Consistent page layouts
- `SpiritualNavigation` - Navigation with spiritual elements
- `SacredFooter` - Footer with spiritual quotes/wisdom

### Form Components

- `DharmaInput` - Styled form inputs
- `SpiritualSelect` - Dropdown with dharmic styling
- `MeditationForm` - Forms for spiritual practices

## 🔄 Development Workflow

### Adding New Components

1. Create component in `shared/components/`
2. Add TypeScript types in `shared/types/`
3. Update this documentation
4. Test in at least one frontend application
5. Version and publish changes

### Updating Design System

1. Modify `shared/styles/tailwind.dharma.config.js`
2. Test across all frontend applications
3. Update documentation with new design tokens
4. Coordinate rollout across teams

## 📊 Usage Tracking

Components are used across:

- ✅ Brand Website (`Brand_Webpage/`)
- ✅ Chat Application (`dharmamind-chat/`)
- ✅ Community Platform (`DhramaMind_Community/`)

## 🛠️ Maintenance

### Regular Tasks

- Keep components updated with latest React patterns
- Ensure accessibility compliance
- Update design tokens as brand evolves
- Maintain TypeScript definitions

### Breaking Changes

- Document breaking changes in CHANGELOG
- Provide migration guides
- Support gradual migration across applications
- Version components appropriately

---

For questions about shared components, contact the frontend architecture team or create an issue in the repository.
