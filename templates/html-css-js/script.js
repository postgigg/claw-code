// Dark mode toggle
const themeToggle = document.getElementById('theme-toggle');
const themeIcon = document.getElementById('theme-icon');

function setTheme(dark) {
  document.documentElement.classList.toggle('dark', dark);
  themeIcon.textContent = dark ? '☀️' : '🌙';
  localStorage.setItem('theme', dark ? 'dark' : 'light');
}

// Check saved preference or system preference
const savedTheme = localStorage.getItem('theme');
if (savedTheme === 'dark' || (!savedTheme && window.matchMedia('(prefers-color-scheme: dark)').matches)) {
  setTheme(true);
}

themeToggle.addEventListener('click', () => {
  const isDark = document.documentElement.classList.contains('dark');
  setTheme(!isDark);
});

// Mobile menu toggle
const mobileMenuBtn = document.getElementById('mobile-menu-btn');
const mobileMenu = document.getElementById('mobile-menu');

mobileMenuBtn.addEventListener('click', () => {
  mobileMenu.classList.toggle('hidden');
});

// Close mobile menu when clicking a link
mobileMenu.querySelectorAll('a').forEach(link => {
  link.addEventListener('click', () => {
    mobileMenu.classList.add('hidden');
  });
});

// Contact form handling
const contactForm = document.getElementById('contact-form');
const formStatus = document.getElementById('form-status');

contactForm.addEventListener('submit', (e) => {
  e.preventDefault();

  const formData = new FormData(contactForm);
  const data = Object.fromEntries(formData);

  // TODO: Replace with your form handling logic
  // Options: Formspree, Netlify Forms, custom API endpoint
  console.log('Form submitted:', data);

  formStatus.textContent = 'Message sent! We\'ll get back to you soon.';
  formStatus.classList.remove('hidden', 'text-red-500');
  formStatus.classList.add('text-green-600');
  contactForm.reset();

  setTimeout(() => {
    formStatus.classList.add('hidden');
  }, 5000);
});

// Scroll-based nav background
window.addEventListener('scroll', () => {
  const nav = document.querySelector('nav');
  if (window.scrollY > 10) {
    nav.classList.add('shadow-sm');
  } else {
    nav.classList.remove('shadow-sm');
  }
});
