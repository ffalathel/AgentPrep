# Security Policy

## Supported Versions

We currently support the following versions with security updates:

| Version | Supported          |
| ------- | ------------------ |
| 0.1.x   | :white_check_mark: |

## Reporting a Vulnerability

If you discover a security vulnerability, please **do not** open a public issue. Instead, please report it via one of the following methods:

1. **Email**: [Add your security email address]
2. **Private Security Advisory**: [Add link to GitHub Security Advisory if applicable]

Please include the following information in your report:

- Description of the vulnerability
- Steps to reproduce the issue
- Potential impact
- Suggested fix (if any)

We will acknowledge receipt of your report within 48 hours and provide an update on the status of the vulnerability within 7 days.

## Security Best Practices

### API Keys and Secrets

**Never commit API keys or secrets to the repository.** AgentPrep uses environment variables for all sensitive configuration:

- `OPENAI_API_KEY` - OpenAI API key (optional)
- `ANTHROPIC_API_KEY` - Anthropic API key (optional)
- `GEMINI_API_KEY` - Google Gemini API key (optional)

Always use environment variables or secure secret management systems. The `.env` file is gitignored for your protection.

### Data Privacy

- AgentPrep processes data locally and does not send data to external services unless you explicitly configure LLM providers
- When using LLM providers, be aware that prompts (which may include data summaries) are sent to third-party APIs
- Review your data handling policies before using AgentPrep with sensitive datasets

### Dependencies

We regularly update dependencies to address security vulnerabilities. To update:

```bash
pip install --upgrade -r requirements.txt
```

### Input Validation

AgentPrep uses Pydantic for input validation to prevent injection attacks and ensure data integrity. Always validate user inputs before processing.

## Security Checklist for Contributors

- [ ] No hardcoded secrets or API keys
- [ ] All user inputs are validated
- [ ] Dependencies are up to date
- [ ] No sensitive data in logs or error messages
- [ ] Environment variables are used for configuration
- [ ] File paths are validated to prevent directory traversal

## Disclosure Policy

When a security vulnerability is fixed, we will:

1. Release a security update as soon as possible
2. Document the vulnerability and fix in the release notes
3. Credit the reporter (if they wish to be credited)

Thank you for helping keep AgentPrep secure!
