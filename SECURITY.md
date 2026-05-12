# Security Policy

## Project Status

ScholarFlux MCP is currently in **beta** (v0.1.0). While we remain committed to security and will address vulnerabilities as they become known, please be aware:

- This is pre-release software under active development
- APIs and interfaces may change between versions
- Security patches will be incorporated as vulnerabilities are discovered
- We encourage security researchers to help us identify and fix issues

Starting from version v0.1.0, we will release patches for security vulnerabilities as they are reported.

**Note:** As we move toward a stable 1.0 release, we will establish a more formal security support timeline.

## Reporting a Vulnerability

**Please do not report security vulnerabilities through public GitHub issues.**

We take security seriously, even during beta. Please report security vulnerabilities by:

1. **Email:** scholar.flux@gmail.com
2. **GitHub Security Advisories:** Use the "Security" tab in this repository

We aim to respond within 72 hours. As this is a beta project under active development, response times may vary, but we are committed to addressing security concerns promptly.

Please include the following information in your report:
- Type of vulnerability
- Full paths of source file(s) related to the vulnerability
- Location of the affected source code (tag/branch/commit or direct URL)
- Step-by-step instructions to reproduce the issue
- Proof-of-concept or exploit code (if possible)
- Impact of the vulnerability, including how an attacker might exploit it

**Beta Disclosure:** Given the pre-release status, we may address critical vulnerabilities immediately in the main branch. Less critical issues will be tracked and resolved in subsequent releases.

## Security Considerations

### API Keys and Credentials

The underlying ScholarFlux library interacts with various academic databases and APIs that may require authentication:

- **Never hardcode API keys** in your code or MCP Server configuration. Ensure that sensitive data isn't committed to version control after importing and using the code directly
- When interfacing with agents via an MCP server configuration, never type an API key into the prompt - API keys should be handled via environment variables only
- Use environment variables or secure credential management systems
- Rotate API keys regularly
- Use read-only or minimal-privilege API keys when possible

**Example MCP Server configuration:**
```json
{  
    "mcpServers": {  
      "scholar-flux": {  
        "args": [  
          "-m",  
          "scholar_flux_mcp"  
        ],  
        "command": "python",  
        "disabled": false,  
        "env": {  
          "CORE_API_KEY": "${CORE_API_KEY}",  
          "OLLAMA_API_KEY": "${OLLAMA_API_KEY}",  
          "PUBMED_API_KEY": "${PUBMED_API_KEY}",  
          "SPRINGER_NATURE_API_KEY": "${SPRINGER_NATURE_API_KEY}",
          "OPENAI_API_KEY": "${OPENAI_API_KEY}",  
          "SCHOLAR_FLUX_DEFAULT_MAILTO": "${SCHOLAR_FLUX_DEFAULT_MAILTO}",  
          "SCHOLAR_FLUX_DEFAULT_USER_AGENT": "${SCHOLAR_FLUX_DEFAULT_USER_AGENT}",  
          "SCHOLAR_FLUX_DEFAULT_RESPONSE_CACHE_STORAGE": "redis",  
          "SCHOLAR_FLUX_DEFAULT_SESSION_CACHE_BACKEND": "redis",  
          "SCHOLAR_FLUX_MCP_DEFAULT_EMBEDDING_MODEL_PROVIDER": "ollama",  
          "SCHOLAR_FLUX_MCP_DEFAULT_MODEL_PROVIDER": "ollama_cloud",  
          "SCHOLAR_FLUX_MCP_OLLAMA_CLOUD_MODEL": "minimax-m2.5:cloud",  
          "SCHOLAR_FLUX_MCP_OLLAMA_MODEL": "glm-4.7-flash:latest"  
        },  
        "timeout": 300000  
      }  
    }  
  }

```

### Internal Secret Management

The internal ScholarFlux implements security features inherited by the MCP server to guard API keys from accidental exposure.

- **ConfigLoader** wraps API keys in `SecretStr` objects—they won't leak via `repr()` or `str()`
- **SensitiveDataMasker** pattern-matches API keys, database URIs, and private keys before they hit logs
- **MaskingFilter** scrubs any remaining sensitive strings from log output

Even if you accidentally log a config object, credentials stay masked:

```python
import logging

from scholar_flux_mcp.agents import PydanticAIModelFactory

logger = logging.getLogger('scholar_flux')
logger.setLevel(logging.DEBUG)


# Even if you log the entire config, api keys are masked
claude_model = PydanticAIModelFactory.create("anthropic")
logger.debug((claude_model.client.__dict__))
# Output: {'api_key': '***', 'auth_token': None, '_version': '0.77.0', '_base_url': 'https://api.anthropic.com', ...'}
```
See the ScholarFlux [SECURITY.md](https://github.com/SammieH21/scholar-flux/blob/main/SECURITY.md) for details

### Caching Security

ScholarFlux uses `requests-cache` with security features:

- Cached responses may contain sensitive data
- Consider cache expiration policies for sensitive queries through environment variable configuration (e.g., SCHOLAR_FLUX_DEFAULT_SESSION_CACHE_TTL)
- Be mindful of cached credentials in shared environments

See the ScholarFlux [SECURITY.md](https://github.com/SammieH21/scholar-flux/blob/main/SECURITY.md) for details


### Security Notes:

**Never commit encryption keys to version control**
Rotate encryption keys periodically
If the encryption key is lost, cached data cannot be recovered
Use different keys for development and production environments
Consider key management systems (AWS KMS, HashiCorp Vault) for production

### Database Connections

When configuring the ScholarFlux session cache or response cache storage for use via environment variables (SQLAlchemy, Redis, MongoDB):

- **Never use default credentials** in production
- Use connection string encryption
- Implement proper authentication and authorization
- Use TLS/SSL for database connections
- Follow the principle of least privilege for database users
- Regularly update database drivers

### Input Validation

ScholarFlux and the ScholarFlux MCP server uses Pydantic for data validation:

- All user inputs are validated before processing
- API responses are parsed and validated
- Type checking prevents injection attacks

### Dependency Security

We regularly monitor and update dependencies:

- All dependencies are tracked in `poetry.lock`
- We use dependencies such as `mcp`, `pydantic_ai`, and `rapidfuzz` with security best practices in mind
- Run `poetry update` regularly to get security patches
- Monitor GitHub Security Advisories for this repository

## Best Practices for Users

### Beta Software Notice
As the ScholarFlux MCP server is in beta:
- Test thoroughly in development environments before production use
- Monitor the repository for updates and security advisories
- Report any security concerns you discover
- Stay updated with the latest beta releases
- Understand that breaking changes may occur between beta versions

### 1. Keep Dependencies Updated
```bash
poetry update
```

### 2. Use Virtual Environments
Always use isolated environments to prevent dependency conflicts:
```bash
poetry install
poetry shell
```

### 3. Minimal Installations
Only install extras you need:
```bash
# Only install what you use. For basic development:
poetry install --with dev,testing
```

## Known Security Limitations

### Third-Party API Dependencies
- ScholarFlux MCP relies on external academic APIs
- Security of data depends on third-party providers
- API availability and authentication methods may change
- Users are responsible for complying with API terms of service

### Data Privacy
- Scholarly data may contain personal information
- Users must comply with relevant data protection regulations (GDPR, CCPA, etc.)
- Be mindful of caching personally identifiable information
- Consider data retention policies

## Security Updates

As a beta project, we are committed to:
- Responding to reported vulnerabilities within 72 hours
- Incorporating security patches into subsequent releases
- Addressing critical vulnerabilities as quickly as possible
- Crediting security researchers (unless they prefer to remain anonymous)
- Maintaining transparency about known security issues

**Development Timeline:**
- **Beta (current):** Security fixes incorporated as vulnerabilities are discovered
- **Stable 1.0+:** Formal security advisory system and regular patch schedule

We appreciate the security community's patience and collaboration as we work toward a stable release.

## Responsible Disclosure

We follow a coordinated disclosure policy adapted for beta software:
1. Security researchers report vulnerabilities privately
2. We acknowledge receipt within 72 hours
3. We work with researchers to understand and validate the issue
4. We develop and test a fix
5. For critical vulnerabilities: immediate patch to main branch
6. For non-critical issues: inclusion in next release with security notes
7. We publicly credit the researcher (with their permission)

**Beta Consideration:** Given the active development nature of this project, fixes may be deployed more rapidly than in stable software, and we may coordinate disclosure timing based on severity and fix complexity.

## Security Resources

- [OWASP Top 10](https://owasp.org/www-project-top-ten/)
- [Python Security Best Practices](https://python-security.readthedocs.io/vulnerabilities.html)
- [Requests Security](https://requests.readthedocs.io/en/latest/user/advanced/#ssl-cert-verification)

## Contact

For security concerns, please contact:
- **Primary:** Use GitHub Security Advisories (Security tab in this repository)
- **Alternative:** Open a private discussion or contact the maintainers
- **Public discussions:** Only for non-sensitive security topics

---

**Note:** This security policy reflects our commitment to security during beta development and is subject to change as the project matures. Please check back regularly for updates.
