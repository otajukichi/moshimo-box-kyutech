# Episodes

Add one `.yaml` file per episode. Files are validated at application startup.
Use an existing episode as the template. Keep copyrighted character names and
real-person names generalized.

Formal mode requires all of the following:

```yaml
formal_mode_allowed: true
public_demo_allowed: true
limited_only: false
```

Setting `enabled: false` keeps an episode in Git without making it selectable.
